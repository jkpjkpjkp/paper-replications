"""Experimental PostgreSQL-aggregate TAQ worker; direct worker benchmarked faster.

WRDS-side daily TAQ-to-minute worker, loaded into a dedicated Jupyter kernel.

Requires WRDS_USERNAME in the loading kernel. No raw trade files are written.
PostgreSQL groups source trades; only minute bars stream through a bounded pipe.
One daily Parquet is retained until acknowledge() is called after local checksum
verification. Time boundaries use SIP time_m in America/New_York local time.

start_day(date, mappings, close_time='16:00:00', price_condition_sql=None,
          volume_condition_sql=None)

The optional SQL expressions are evaluated against source table alias ``t``.
They are provenance-bearing caller choices, not claimed paper conventions.
Canonical OHLC uses price eligibility; amount/volume/trade_count use volume
eligibility. raw_* OHLC preserves all volume-eligible prints, and eligible_*
amount/volume/trade_count permits an internally consistent alternative.
Default predicates are TRUE (no sale-condition exclusion), explicitly recorded.
"""
import datetime
import hashlib
import os
import re
import shutil
import threading
import time
import uuid
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
import wrds
from psycopg2 import sql


db = wrds.Connection(wrds_username=WRDS_USERNAME)
db.connection.close()
export_root = Path.cwd() / ('wrds_song26l_taq_pg_' + uuid.uuid4().hex)
export_root.mkdir(mode=0o700)
_lock = threading.RLock()
_state = {'state': 'idle'}
_thread = None
_active_connection = None
_closed = False


def query(statement, params=None):
    """Small metadata queries; do not fetch raw TAQ trades with this helper."""
    conn = db.engine.raw_connection()
    try:
        with conn.cursor() as cur:
            cur.execute('SET TRANSACTION READ ONLY')
            cur.execute(statement, params)
            keys = [x.name for x in cur.description]
            return [dict(zip(keys, row)) for row in cur.fetchall()]
    finally:
        conn.rollback()
        conn.close()


def _predicate(value):
    if value is None:
        return 'TRUE'
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Eligibility must be a nonempty SQL expression or None')
    if any(x in value for x in (';', '--', '/*', '*/', '\x00')):
        raise ValueError('Eligibility must be one SQL expression without comments')
    return value.strip()


def _normalise_mappings(mappings):
    if not isinstance(mappings, list) or not mappings:
        raise ValueError('mappings must be a nonempty list of mapping records')
    result, seen = [], {}
    for item in mappings:
        permno = int(item['permno'])
        symbol = str(item['sym_root']).strip()
        suffix = str(item.get('sym_suffix') or '').strip()
        if permno <= 0 or not symbol or len(symbol) > 30 or len(suffix) > 30:
            raise ValueError('Invalid PERMNO or TAQ symbol')
        key = (symbol, suffix)
        if key in seen:
            if seen[key] != permno:
                raise ValueError('One TAQ symbol maps to multiple PERMNOs: ' + repr(key))
            continue
        seen[key] = permno
        result.append((permno, symbol, suffix))
    return sorted(result)


def _statement(day, mappings, close_time, price_condition, volume_condition):
    values = sql.SQL(',').join(sql.SQL('({},{},{})').format(
        sql.Literal(p), sql.Literal(r), sql.Literal(s)) for p, r, s in mappings)
    # Hash aggregates retain minute groups rather than sorting/returning trades.
    # The second aggregate merges any same-day aliases of the same PERMNO.
    statement = sql.SQL("""
      WITH base AS (
        SELECT m.permno::integer AS permno,
          m.sym_root::text AS sym_root, m.sym_suffix::text AS sym_suffix,
          floor(extract(epoch FROM t.time_m)/60)::smallint AS minute_of_day,
          (extract(epoch FROM t.time_m)*1000000)::bigint AS time_us,
          coalesce(t.tr_seqnum,0)::bigint AS seq,
          t.price::double precision AS price, t.size::bigint AS size,
          ({price_condition})::boolean AS eligible,
          ARRAY[
            (extract(epoch FROM t.time_m)*1000000000+coalesce(t.time_m_nano,0))::double precision,
            coalesce(t.tr_seqnum,0)::double precision,
            CASE WHEN coalesce(t.ex,'')='' THEN 0 ELSE ascii(t.ex) END::double precision,
            t.price::double precision,t.size::double precision] AS ord
        FROM {source} AS t
        JOIN (VALUES {values}) AS m(permno,sym_root,sym_suffix)
          ON t.sym_root=m.sym_root AND coalesce(t.sym_suffix,'')=m.sym_suffix
        WHERE t.time_m >= TIME '09:30:00' AND t.time_m < {close_time}::time
          AND t.price > 0 AND t.size > 0 AND t.tr_corr IN ('00','01')
          AND ({volume_condition})
      ), by_symbol AS (
        SELECT permno,sym_root,sym_suffix,minute_of_day,
          min(ord) AS raw_open_key,max(ord) AS raw_close_key,
          min(ord) FILTER (WHERE eligible) AS open_key,
          max(ord) FILTER (WHERE eligible) AS close_key,
          max(price) FILTER (WHERE eligible) AS high,
          min(price) FILTER (WHERE eligible) AS low,
          max(price) AS raw_high,min(price) AS raw_low,
          sum(price*size)::double precision AS amount,
          sum(size)::bigint AS volume,count(*)::bigint AS trade_count,
          sum(price*size) FILTER (WHERE eligible)::double precision AS eligible_amount,
          sum(size) FILTER (WHERE eligible)::bigint AS eligible_volume,
          count(*) FILTER (WHERE eligible)::bigint AS eligible_trade_count,
          min(time_us)::bigint AS first_sip_time_us,
          max(time_us)::bigint AS last_sip_time_us,
          max(abs(seq)) AS max_sequence_number
        FROM base GROUP BY permno,sym_root,sym_suffix,minute_of_day
      ), bars AS (
        SELECT permno,minute_of_day,min(sym_root) AS sym_root,min(sym_suffix) AS sym_suffix,
          count(*)::integer AS taq_symbol_count,
          min(open_key) AS open_key,max(close_key) AS close_key,
          max(high) AS high,min(low) AS low,
          min(raw_open_key) AS raw_open_key,max(raw_close_key) AS raw_close_key,
          max(raw_high) AS raw_high,min(raw_low) AS raw_low,
          sum(amount)::double precision AS amount,
          sum(volume)::bigint AS volume,sum(trade_count)::bigint AS trade_count,
          sum(eligible_amount)::double precision AS eligible_amount,
          sum(eligible_volume)::bigint AS eligible_volume,
          sum(eligible_trade_count)::bigint AS eligible_trade_count,
          min(first_sip_time_us)::bigint AS first_sip_time_us,
          max(last_sip_time_us)::bigint AS last_sip_time_us,
          max(max_sequence_number)::bigint AS max_sequence_number
        FROM by_symbol GROUP BY permno,minute_of_day
      )
      SELECT {day}::date AS date,permno,sym_root,sym_suffix,taq_symbol_count,
        (minute_of_day-570)::smallint AS minute_index,
        {day}::date+minute_of_day*INTERVAL '1 minute' AS minute_start,
        open_key[4] AS open,high,low,close_key[4] AS close,amount,volume,trade_count,
        raw_open_key[4] AS raw_open,raw_high,raw_low,raw_close_key[4] AS raw_close,
        eligible_amount,eligible_volume,eligible_trade_count,first_sip_time_us,last_sip_time_us,
        max_sequence_number
      FROM bars ORDER BY permno,minute_of_day
    """).format(source=sql.Identifier('taqm_' + str(day.year), 'ctm_' + day.strftime('%Y%m%d')),
        values=values,day=sql.Literal(day.isoformat()),close_time=sql.Literal(close_time),
        price_condition=sql.SQL(price_condition),volume_condition=sql.SQL(volume_condition))
    return sql.SQL("COPY ({}) TO STDOUT WITH (FORMAT CSV, HEADER TRUE, NULL '\\N', ENCODING 'UTF8')").format(statement)


_BAR_TYPES = {
    'date':pa.date32(),'permno':pa.int32(),'sym_root':pa.string(),'sym_suffix':pa.string(),
    'taq_symbol_count':pa.int32(),'minute_index':pa.int16(),'minute_start':pa.timestamp('us'),
    **{k:pa.float64() for k in ['open','high','low','close','amount','raw_open','raw_high',
                                'raw_low','raw_close','eligible_amount']},
    **{k:pa.int64() for k in ['volume','trade_count','eligible_volume','eligible_trade_count',
                             'first_sip_time_us','last_sip_time_us','max_sequence_number']},
}


def _run_day(day, mappings, close_time, price_condition, volume_condition):
    global _active_connection
    conn, local_db, source, producer = None, None, None, None
    read_fd, write_fd = None, None
    copied = {}
    path = export_root / (day.isoformat() + '.parquet')
    temporary_path = export_root / (day.isoformat() + '.partial.parquet')
    temp_dir = export_root / 'duckdb_tmp'
    started = time.monotonic()
    try:
        conn = db.engine.raw_connection()
        with _lock:
            _active_connection = conn
        read_fd, write_fd = os.pipe()
        copy = _statement(day, mappings, close_time, price_condition, volume_condition)

        def produce():
            # This thread owns both the DB cursor and pipe's write descriptor.
            try:
                with os.fdopen(write_fd, 'wb') as sink, conn.cursor() as cur:
                    cur.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
                    cur.execute("SET LOCAL DateStyle = 'ISO, YMD'")
                    cur.execute('SET LOCAL max_parallel_workers_per_gather = 0')
                    cur.execute("SET LOCAL work_mem='128MB'")
                    cur.copy_expert(copy.as_string(cur), sink, size=1024 * 1024)
                    copied['rows'] = cur.rowcount
            except BaseException as exc:
                copied['error'] = str(exc)

        producer = threading.Thread(target=produce, name='song26l-copy', daemon=True)
        producer.start()
        source = os.fdopen(read_fd, 'rb')
        read_fd = None
        try:
            reader = pacsv.open_csv(source,
                read_options=pacsv.ReadOptions(block_size=8 * 1024**2, use_threads=False),
                parse_options=pacsv.ParseOptions(newlines_in_values=True, ignore_empty_lines=False),
                convert_options=pacsv.ConvertOptions(column_types=_BAR_TYPES,
                    null_values=['\\N'], strings_can_be_null=True,
                    true_values=['t','true','TRUE','1'],
                    false_values=['f','false','FALSE','0'],
                    quoted_strings_can_be_null=False))
        except pa.ArrowInvalid:
            producer.join(timeout=30)
            if copied.get('error'):
                raise RuntimeError(copied['error'])
            raise
        bars = reader.read_all()
        max_sequence_number = max(bars.column('max_sequence_number').to_pylist(),default=0)
        if max_sequence_number > 9007199254740991:
            raise RuntimeError('Sequence number too large for exact float8 sort key')
        bars = bars.drop(['max_sequence_number'])
        local_db = duckdb.connect()
        local_db.execute("SET memory_limit='512MB'")
        local_db.execute('SET threads=1')
        source.close()
        source = None
        producer.join(timeout=30)
        if producer.is_alive():
            raise RuntimeError('COPY producer did not finish after Arrow input was consumed')
        if copied.get('error'):
            raise RuntimeError(copied['error'])
        metadata = dict(bars.schema.metadata or {})
        metadata.update({
            b'source': ('taqm_' + str(day.year) + '.ctm_' + day.strftime('%Y%m%d')).encode(),
            b'timezone': b'America/New_York',
            b'minute_convention': b'left-closed right-open; minute_start is NY local naive time',
            b'session_open': b'09:30:00', b'session_close_exclusive': close_time.encode(),
            b'price_condition_sql': price_condition.encode(),
            b'volume_condition_sql': volume_condition.encode(),
            b'base_filter': b'price>0 AND size>0 AND tr_corr IN (00,01)',
            b'ohlc': b'price-eligible prints; raw_* OHLC uses all volume-eligible prints',
            b'volume': b'amount,volume,trade_count include all volume-eligible prints',
            b'numerics': b'prices and dollar amount float64; volume/count int64',
            b'missing_minutes': b'absent; no filling or synthetic prices',
            b'corporate_adjustment': b'none; original nominal TAQ prices',
            b'aggregation_engine': b'PostgreSQL17 min/max(float8[]) hash aggregation',
            b'ohlc_order': b'SIP nanoseconds,sequence,exchange ASCII,price,size; tr_id omitted',
        })
        bars = bars.replace_schema_metadata(metadata)
        trade_count = sum(bars.column('trade_count').to_pylist())
        if copied.get('rows', -1) != bars.num_rows:
            raise RuntimeError('COPY bar rows differs from parsed bar rows')
        pq.write_table(bars, temporary_path, compression='zstd', compression_level=3,
                       row_group_size=100000, version='2.6')
        temporary_path.replace(path)
        with path.open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        # Register the small bar table only after the streaming query finished.
        local_db.register('bars_summary', bars)
        names = ['rows','securities','volume','trades','amount','price_eligible_trades',
                 'minutes_without_eligible_price','minutes_with_multiple_symbols']
        row = local_db.execute('''SELECT count(*), count(DISTINCT permno),
            sum(volume)::BIGINT, sum(trade_count)::BIGINT, sum(amount)::DOUBLE,
            sum(eligible_trade_count)::BIGINT,
            count(*) FILTER (WHERE close IS NULL),
            count(*) FILTER (WHERE taq_symbol_count>1) FROM bars_summary''').fetchone()
        totals = dict(zip(names, row))
        observed = set(bars.column('permno').to_pylist())
        result = {'state': 'complete', 'date': day.isoformat(),
            'path': export_root.name + '/' + path.name, 'name': path.name,
            'format': 'parquet', 'bytes': path.stat().st_size, 'sha256': digest,
            'copy_rows': trade_count, 'source_bar_rows': copied.get('rows'), 'totals': totals,
            'max_sequence_number': max_sequence_number, 'aggregation_engine': 'postgresql',
            'mapped_securities': len(set(m[0] for m in mappings)),
            'mapped_symbols': len(mappings),
            'missing_permnos': sorted(set(m[0] for m in mappings) - observed),
            'close_time_exclusive': close_time, 'price_condition_sql': price_condition,
            'volume_condition_sql': volume_condition,
            'elapsed_seconds': round(time.monotonic() - started, 3),
            'completed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
        with _lock:
            _state.clear()
            _state.update(result)
    except BaseException as exc:
        with _lock:
            _state.update(state='error', error=str(exc), error_type=type(exc).__name__,
                elapsed_seconds=round(time.monotonic() - started, 3))
        temporary_path.unlink(missing_ok=True)
        path.unlink(missing_ok=True)
    finally:
        if source is not None:
            source.close()
        if read_fd is not None:
            os.close(read_fd)
        if producer is not None and producer.is_alive() and conn is not None:
            try:
                conn.cancel()
            except Exception:
                pass
            producer.join(timeout=30)
        if conn is not None:
            try:
                conn.rollback()
            finally:
                conn.close()
        if local_db is not None:
            local_db.close()
        shutil.rmtree(temp_dir, ignore_errors=True)
        with _lock:
            _active_connection = None


def start_day(date, mappings, close_time='16:00:00', price_condition_sql=None,
              volume_condition_sql=None):
    global _thread
    day = datetime.date.fromisoformat(str(date))
    close = datetime.time.fromisoformat(close_time)
    if not datetime.time(9,30) < close <= datetime.time(16):
        raise ValueError('close_time must be after 09:30 and at or before 16:00 NY local')
    close_time = close.isoformat()
    mappings = _normalise_mappings(mappings)
    price_condition = _predicate(price_condition_sql)
    volume_condition = _predicate(volume_condition_sql)
    with _lock:
        if _closed:
            raise RuntimeError('Worker has been cleaned up')
        if _thread is not None and _thread.is_alive():
            raise RuntimeError('A daily export is already running')
        if _state.get('state') == 'complete':
            raise RuntimeError('Download and verify completed file, then acknowledge it first')
        _state.clear()
        _state.update(state='running', date=day.isoformat(),
            started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            started_monotonic=time.monotonic(), price_condition_sql=price_condition,
            volume_condition_sql=volume_condition, close_time_exclusive=close_time)
        _thread = threading.Thread(target=_run_day,
            args=(day,mappings,close_time,price_condition,volume_condition),
            name='song26l-day', daemon=True)
        _thread.start()
        return {'started': True, 'date': day.isoformat(), 'state': 'running'}


def status():
    with _lock:
        result = dict(_state)
        monotonic = result.pop('started_monotonic', None)
        if monotonic is not None:
            result['elapsed_seconds'] = round(time.monotonic() - monotonic, 3)
        if result.get('state') in ('complete','error') and _thread is not None and _thread.is_alive():
            result['state'] = 'running'
            result['stage'] = 'finalizing'
        if result.get('state') == 'running':
            result['staged_bytes'] = sum(p.stat().st_size for p in export_root.rglob('*') if p.is_file())
        return result


def acknowledge(sha256=None):
    """Delete the remote daily file after the caller verifies its local copy."""
    with _lock:
        if _thread is not None and _thread.is_alive():
            raise RuntimeError('Export thread has not finished')
        if _state.get('state') != 'complete':
            raise RuntimeError('No completed daily export to acknowledge')
        if sha256 is not None and sha256 != _state['sha256']:
            raise ValueError('Acknowledgement checksum does not match export')
        day = _state['date']
        (export_root / _state['name']).unlink()
        _state.clear()
        _state.update(state='idle', acknowledged_date=day)
        return dict(_state)


def cleanup():
    global _closed
    with _lock:
        if _thread is not None and _thread.is_alive():
            return {'cleaned': False, 'reason': 'daily export running'}
        db.close()
        shutil.rmtree(export_root, ignore_errors=True)
        _closed = True
        return {'cleaned': True}
