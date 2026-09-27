//! PROBE-ONLY (tui-scroll-retain2 lane, branch lane/hillclimb-tui-scroll-probe2):
//! a kick-file-triggered census of the transcript state the TUI retains at
//! the packed-storage tip — the WAVE-5 REMAINING retention classes: the
//! packed per-span records (style-table + offset-redundancy sizing), the
//! transient render churn (uncacheable + first-visit paths), and the
//! freed-page straggler attribution. Never ships: this file exists to
//! decompose the retained heap BEFORE any product edit (baseline first).
#![allow(dead_code)]

use super::AgentView;
use crate::chat::ChatEntry;
use serde_json::json;
use std::cell::Cell;
use std::io::Write;

/// Transient-path churn counters (probe): [visits, rows, spans,
/// content_bytes] for uncacheable (transient) entry renders and for
/// cacheable first-visit renders, [renders, rows, content_bytes] for
/// transcript-tail renders, and splash renders. Absolute since process
/// start; census rows are append-only, so phase deltas subtract offline.
std::thread_local! {
    pub(super) static PROBE_UNC: Cell<[u64; 4]> = const { Cell::new([0; 4]) };
    pub(super) static PROBE_FV: Cell<[u64; 4]> = const { Cell::new([0; 4]) };
    pub(super) static PROBE_TAIL: Cell<[u64; 3]> = const { Cell::new([0; 3]) };
    pub(super) static PROBE_SPLASH: Cell<u64> = const { Cell::new(0) };
}

/// Note one uncacheable (transient) entry render (probe).
pub(super) fn note_unc(rows: usize, spans: usize, content: usize) {
    PROBE_UNC.with(|c| {
        let mut v = c.get();
        v[0] += 1;
        v[1] += rows as u64;
        v[2] += spans as u64;
        v[3] += content as u64;
        c.set(v);
    });
}

/// Note one cacheable first-visit entry render (probe).
pub(super) fn note_fv(rows: usize, spans: usize, content: usize) {
    PROBE_FV.with(|c| {
        let mut v = c.get();
        v[0] += 1;
        v[1] += rows as u64;
        v[2] += spans as u64;
        v[3] += content as u64;
        c.set(v);
    });
}

/// Note one transcript-tail render (probe).
pub(super) fn note_tail(rows: usize, content: usize) {
    PROBE_TAIL.with(|c| {
        let mut v = c.get();
        v[0] += 1;
        v[1] += rows as u64;
        v[2] += content as u64;
        c.set(v);
    });
}

/// Note one splash render (probe).
pub(super) fn note_splash() {
    PROBE_SPLASH.with(|c| c.set(c.get() + 1));
}

/// Packed-slot census stats (probe): totals plus the style-table and
/// span-length distributions that size a denser record encoding.
#[derive(Default, Clone)]
struct SlotStats {
    slots: usize,
    rows: usize,
    spans: usize,
    records_as_built_bytes: usize,
    blob_cap: usize,
    index_cap: usize,
    /// Per-slot distinct styles: histogram over the distinct count.
    distinct_styles: std::collections::BTreeMap<usize, usize>,
    max_distinct_styles: usize,
    style_table_bytes: usize,
    /// Span-content length histogram (varint sizing): [<=7, <=15, <=31,
    /// <=63, <=127, <=255, <=1023, <=65535, >65535].
    len_hist: [usize; 9],
    total_len_bytes: usize,
    varint_len_bytes: usize,
    /// Projected packed bytes under denser encodings (records only;
    /// +4B/row for the per-row blob-offset array is added separately).
    rec8_bytes: usize,
    rec6_bytes: usize,
    rec_varlen_u32style_bytes: usize,
    per_kind: std::collections::BTreeMap<&'static str, SlotKindStats>,
}

#[derive(Default, Clone)]
struct SlotKindStats {
    slots: usize,
    rows: usize,
    spans: usize,
    records_as_built_bytes: usize,
    blob_cap: usize,
}

fn varint_len(len: u32) -> usize {
    match len {
        0..=0x7f => 1,
        0x80..=0x3fff => 2,
        0x4000..=0x1f_ffff => 3,
        0x20_0000..=0x0fff_ffff => 4,
        _ => 5,
    }
}

fn entry_kind(entry: &ChatEntry) -> &'static str {
    match entry {
        ChatEntry::Status { .. } => "status",
        ChatEntry::User { .. } => "user",
        ChatEntry::SlashCommand { .. } => "slash",
        ChatEntry::CompactionSummary { .. } => "compaction",
        ChatEntry::Assistant(_) => "assistant",
        ChatEntry::Tool(_) => "tool",
        ChatEntry::AgentMessage(_) => "agent_message",
        ChatEntry::SkillInvocation(_) => "skill_invocation",
        ChatEntry::InjectedPrompt(_) => "injected_prompt",
        ChatEntry::BashExecution(_) => "bash_execution",
        ChatEntry::ShellCompletion(_) => "shell_completion",
        ChatEntry::RefinementOutcome(_) => "refinement_outcome",
        ChatEntry::CustomPanel(_) => "custom_panel",
    }
}

fn value_bytes(value: &serde_json::Value) -> usize {
    match value {
        serde_json::Value::String(s) => s.len(),
        serde_json::Value::Number(_) | serde_json::Value::Bool(_) => 8,
        serde_json::Value::Null => 0,
        serde_json::Value::Array(items) => 16 + items.iter().map(value_bytes).sum::<usize>(),
        serde_json::Value::Object(map) => {
            16 + map
                .iter()
                .map(|(k, v)| 16 + k.len() + value_bytes(v))
                .sum::<usize>()
        }
    }
}

fn entry_content_bytes(entry: &ChatEntry) -> usize {
    match entry {
        ChatEntry::Status { text, .. }
        | ChatEntry::User { text }
        | ChatEntry::SlashCommand { text } => text.len(),
        ChatEntry::CompactionSummary {
            summary,
            custom_instructions,
            ..
        } => summary.len() + custom_instructions.as_ref().map(|s| s.len()).unwrap_or(0),
        ChatEntry::Assistant(message) => message
            .blocks
            .iter()
            .map(|block| match block {
                crate::chat::MessageBlock::Thinking(text)
                | crate::chat::MessageBlock::Text(text) => 16 + text.len(),
            })
            .sum::<usize>()
            + message.error.as_ref().map(|s| s.len()).unwrap_or(0),
        ChatEntry::Tool(card) => {
            16 + card.id.len() + card.name.len() + value_bytes(&card.args)
                + card
                    .result
                    .as_ref()
                    .map(|result| {
                        16 + result.content.iter().map(value_bytes).sum::<usize>()
                            + value_bytes(&result.details)
                    })
                    .unwrap_or(0)
        }
        other => format!("{other:?}").len(),
    }
}

impl AgentView {
    /// The PROBE census row: the retained packed structure decomposed by
    /// class, the transient churn counters, and the geometry bookkeeping
    /// (heights/sparse/md caches/flush surfaces) that could orphan rows.
    pub(crate) fn census(&self) -> serde_json::Value {
        let mut kinds = std::collections::BTreeMap::new();
        let mut content_bytes = 0usize;
        for entry in &self.chat {
            *kinds.entry(entry_kind(entry)).or_insert(0usize) += 1;
            content_bytes += entry_content_bytes(entry);
        }

        let mut heights_some = [0usize; 3];
        for slots in &self.entry_heights {
            for (detail, slot) in slots.iter().enumerate() {
                if slot.is_some() {
                    heights_some[detail] += 1;
                }
            }
        }
        let mut slots_by_detail = [0usize; 3];
        let mut stats = SlotStats::default();
        let mut per_detail: Vec<SlotStats> = vec![SlotStats::default(); 3];
        let mut layout_no_height = [0usize; 3];
        let mut height_no_layout = [0usize; 3];
        let mut largest_layout: Vec<(usize, usize, usize)> = Vec::new();
        let style_size = std::mem::size_of::<ratatui::style::Style>();
        for (index, slots) in self.entry_layout.iter().enumerate() {
            let kind = self.chat.get(index).map(entry_kind).unwrap_or("?");
            for (detail, slot) in slots.iter().enumerate() {
                let Some(layout) = slot else {
                    if self
                        .entry_heights
                        .get(index)
                        .and_then(|h| h.get(detail))
                        .is_some()
                    {
                        height_no_layout[detail] += 1;
                    }
                    continue;
                };
                slots_by_detail[detail] += 1;
                if self
                    .entry_heights
                    .get(index)
                    .and_then(|h| h.get(detail))
                    .is_none()
                {
                    layout_no_height[detail] += 1;
                }
                let pack = &layout.rows;
                let (rows, spans, index_cap, record_cap, blob_cap) = pack.census_shape();
                let mut distinct: std::collections::HashSet<ratatui::style::Style> =
                    std::collections::HashSet::new();
                let mut len_hist = [0usize; 9];
                let mut total_len_bytes = 0usize;
                let mut varint_bytes = 0usize;
                for (len, style) in pack.census_spans() {
                    distinct.insert(style);
                    let len = len as usize;
                    total_len_bytes += len;
                    varint_bytes += varint_len(len as u32);
                    len_hist[match len {
                        0..=7 => 0,
                        8..=15 => 1,
                        16..=31 => 2,
                        32..=63 => 3,
                        64..=127 => 4,
                        128..=255 => 5,
                        256..=1023 => 6,
                        1024..=65535 => 7,
                        _ => 8,
                    }] += 1;
                }
                let d = distinct.len();
                let slot_stats = |s: &mut SlotStats| {
                    s.slots += 1;
                    s.rows += rows;
                    s.spans += spans;
                    s.records_as_built_bytes += record_cap;
                    s.blob_cap += blob_cap;
                    s.index_cap += index_cap;
                    *s.distinct_styles.entry(d).or_insert(0) += 1;
                    s.max_distinct_styles = s.max_distinct_styles.max(d);
                    s.style_table_bytes += d * style_size;
                    for (i, n) in len_hist.iter().enumerate() {
                        s.len_hist[i] += n;
                    }
                    s.total_len_bytes += total_len_bytes;
                    s.varint_len_bytes += varint_bytes;
                    s.rec8_bytes += spans * 8;
                    s.rec6_bytes += spans * 6;
                    s.rec_varlen_u32style_bytes += varint_bytes + spans * 4;
                };
                slot_stats(&mut stats);
                slot_stats(&mut per_detail[detail]);
                let k = stats.per_kind.entry(kind).or_default();
                k.slots += 1;
                k.rows += rows;
                k.spans += spans;
                k.records_as_built_bytes += record_cap;
                k.blob_cap += blob_cap;
                largest_layout.push((index, detail, blob_cap));
            }
        }
        largest_layout.sort_by_key(|(_, _, bytes)| usize::MAX - *bytes);

        let md = self.md_caches.borrow();
        let mut md_cache_entries = 0usize;
        let mut md_blocks = 0usize;
        let mut md_key_bytes = 0usize;
        let mut md_lines = 0usize;
        let mut md_spans = 0usize;
        let mut md_content_len = 0usize;
        let mut md_content_cap = 0usize;
        for (_index, cache) in md.iter() {
            md_cache_entries += 1;
            for (key, key_lines) in cache.probe_blocks().iter() {
                md_blocks += 1;
                md_key_bytes += key.len();
                for line in key_lines {
                    md_lines += 1;
                    for span in line {
                        md_spans += 1;
                        md_content_len += span.content.len();
                        md_content_cap += span.content.capacity();
                    }
                }
            }
        }

        let flushed_rows = self.flushed_frame.len();
        let flushed_bytes = self.flushed_frame.iter().map(|s| s.len()).sum::<usize>();
        let osc_rows = self.osc_last_rows.len();
        let osc_bytes = self.osc_last_rows.values().map(|s| s.len()).sum::<usize>();

        let unc = PROBE_UNC.with(|c| c.get());
        let fv = PROBE_FV.with(|c| c.get());
        let tail = PROBE_TAIL.with(|c| c.get());
        let splash = PROBE_SPLASH.with(|c| c.get());

        let packed_total_as_built = stats.records_as_built_bytes
            + stats.blob_cap
            + stats.index_cap;
        let packed_total_rec8 = stats.rec8_bytes
            + stats.blob_cap
            + stats.index_cap
            + stats.rows * 4
            + stats.style_table_bytes;
        json!({
            "ts": std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis())
                .unwrap_or(0),
            "sizes": {
                "span": std::mem::size_of::<crate::Span>(),
                "line": std::mem::size_of::<crate::Line>(),
                "style": style_size,
                "packed_span": std::mem::size_of::<crate::view::layout::ProbePackedSpan>(),
            },
            "chat_entries": self.chat.len(),
            "chat_kinds": kinds,
            "chat_content_bytes": content_bytes,
            "layout_slots": stats.slots,
            "layout_slots_by_detail": slots_by_detail,
            "layout_lines": stats.rows,
            "layout_spans": stats.spans,
            "layout_content_len_bytes": stats.total_len_bytes,
            "packed": {
                "records_as_built_bytes": stats.records_as_built_bytes,
                "blob_cap_bytes": stats.blob_cap,
                "index_cap_bytes": stats.index_cap,
                "total_as_built_bytes": packed_total_as_built,
                "row_offset_index_delta_bytes": stats.rows * 4,
                "style_table_bytes": stats.style_table_bytes,
                "distinct_styles_hist": stats.distinct_styles,
                "max_distinct_styles": stats.max_distinct_styles,
                "span_len_hist": {
                    "le7": stats.len_hist[0], "le15": stats.len_hist[1],
                    "le31": stats.len_hist[2], "le63": stats.len_hist[3],
                    "le127": stats.len_hist[4], "le255": stats.len_hist[5],
                    "le1023": stats.len_hist[6], "le65535": stats.len_hist[7],
                    "gt65535": stats.len_hist[8]
                },
                "varint_len_bytes_total": stats.varint_len_bytes,
                "rec8_bytes": stats.rec8_bytes,
                "rec6_bytes": stats.rec6_bytes,
                "rec_varlen_u32style_bytes": stats.rec_varlen_u32style_bytes,
                "projected_total_rec8_bytes": packed_total_rec8
            },
            "layout_per_detail": (0..3).map(|d| json!({
                "slots": per_detail[d].slots,
                "lines": per_detail[d].rows,
                "spans": per_detail[d].spans,
                "records_as_built_bytes": per_detail[d].records_as_built_bytes,
                "blob_cap": per_detail[d].blob_cap,
            })).collect::<Vec<_>>(),
            "layout_per_kind": stats.per_kind.iter().map(|(kind, s)| json!({
                "kind": kind, "slots": s.slots, "lines": s.rows,
                "spans": s.spans,
                "records_as_built_bytes": s.records_as_built_bytes,
                "blob_cap": s.blob_cap,
            })).collect::<Vec<_>>(),
            "layout_top10_by_blob": largest_layout.iter().take(10)
                .map(|(i, d, b)| json!({"entry": i, "detail": d, "blob_bytes": b}))
                .collect::<Vec<_>>(),
            "layout_no_height": layout_no_height,
            "height_no_layout": height_no_layout,
            "heights_some_by_detail": heights_some,
            "transient_counters": {
                "unc_visits": unc[0], "unc_rows": unc[1],
                "unc_spans": unc[2], "unc_content_bytes": unc[3],
                "fv_visits": fv[0], "fv_rows": fv[1],
                "fv_spans": fv[2], "fv_content_bytes": fv[3],
                "tail_renders": tail[0], "tail_rows": tail[1],
                "tail_content_bytes": tail[2],
                "splash_renders": splash,
            },
            "md_cache_entries": md_cache_entries,
            "md_blocks": md_blocks,
            "md_key_bytes": md_key_bytes,
            "md_lines": md_lines,
            "md_spans": md_spans,
            "md_content_len_bytes": md_content_len,
            "md_content_cap_bytes": md_content_cap,
            "sparse_entries": self.sparse_entries.len(),
            "sparse_window": self.sparse_window.is_some(),
            "layout_width": self.layout_width,
            "flushed_frame": {"rows": flushed_rows, "bytes": flushed_bytes},
            "osc_last_rows": {"rows": osc_rows, "bytes": osc_bytes},
        })
    }
}

/// The TUI process's own memory map summary (probe): total RSS, the
/// [heap] mapping, and every anonymous mapping >= 128KB (glibc mmaps
/// large Vec/String allocations directly; they never live in [heap]).
fn self_smaps_summary() -> serde_json::Value {
    let mut rss_kb = 0usize;
    let mut heap_kb = 0usize;
    let mut small_anon_kb = 0usize;
    let mut big_anon: Vec<serde_json::Value> = Vec::new();
    let Ok(text) = std::fs::read_to_string("/proc/self/smaps") else {
        return serde_json::Value::Null;
    };
    let mut in_heap = false;
    let mut is_anon = false;
    let mut map_kb = 0usize;
    let mut map_label = String::new();
    for line in text.lines() {
        let head = line.split_whitespace().collect::<Vec<_>>();
        if head.len() >= 5 && head[0].contains('-') && !head[0].contains(':') {
            if is_anon && map_kb > 0 {
                if map_kb >= 128 {
                    big_anon.push(serde_json::json!({"kb": map_kb, "label": map_label}));
                } else {
                    small_anon_kb += map_kb;
                }
            }
            let path_part = if head.len() > 5 { head[5..].join(" ") } else { String::new() };
            in_heap = path_part.contains("[heap]");
            is_anon = path_part.is_empty() && head[1].contains('p');
            map_kb = 0;
            map_label = head[0].to_string();
        } else if line.starts_with("Rss:") {
            let kb: usize = head[1].parse().unwrap_or(0);
            rss_kb += kb;
            map_kb += kb;
            if in_heap {
                heap_kb += kb;
            }
        }
    }
    if is_anon && map_kb > 0 {
        if map_kb >= 128 {
            big_anon.push(serde_json::json!({"kb": map_kb, "label": map_label}));
        } else {
            small_anon_kb += map_kb;
        }
    }
    serde_json::json!({
        "self_rss_kb": rss_kb,
        "self_heap_kb": heap_kb,
        "self_small_anon_kb": small_anon_kb,
        "self_big_anon_kb": big_anon.iter().map(|m| m["kb"].as_u64().unwrap_or(0)).sum::<u64>(),
        "self_big_anon": big_anon,
    })
}

/// Run the census if a probe request is pending (draw-path, main thread).
/// `<PA_TUI_CENSUS_FILE>.kick` = plain census;
/// `<PA_TUI_CENSUS_FILE>.kick.trim` = trim freed heap first (slack probe).
pub fn maybe_census(view: &AgentView) {
    let Ok(path) = std::env::var("PA_TUI_CENSUS_FILE") else {
        return;
    };
    let plain = format!("{path}.kick");
    let trim = format!("{path}.kick.trim");
    let mode = if std::path::Path::new(&trim).exists() {
        let _ = std::fs::remove_file(&trim);
        pa_types::memory_release::trim_freed_heap();
        Some(true)
    } else if std::path::Path::new(&plain).exists() {
        let _ = std::fs::remove_file(&plain);
        Some(false)
    } else {
        None
    };
    let Some(trimmed) = mode else {
        return;
    };
    let mut census = view.census();
    if let serde_json::Value::Object(map) = &mut census {
        map.insert("trimmed".into(), serde_json::json!(trimmed));
        map.insert("smaps".into(), self_smaps_summary());
    }
    let line = format!("{census}\n");
    let _ = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .and_then(|mut file| file.write_all(line.as_bytes()));
}
