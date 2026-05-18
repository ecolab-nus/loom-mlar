use std::collections::HashMap;

fn extract_module_symbol(line: &str) -> Option<&str> {
    let trimmed = line.trim_start();
    let rest = trimmed.strip_prefix("module @")?;
    let end = rest
        .find(|c: char| c.is_whitespace() || c == '{')
        .unwrap_or(rest.len());
    if end == 0 {
        return None;
    }
    Some(&rest[..end])
}

fn rewrite_module_symbol_line(
    line: &str,
    processor_name_map: &HashMap<String, String>,
) -> Option<String> {
    let module_symbol = extract_module_symbol(line)?;
    let prefixed = processor_name_map.get(module_symbol)?;
    Some(line.replacen(&format!("@{module_symbol}"), &format!("@{prefixed}"), 1))
}

fn rewrite_bind_mem_line(line: &str, memory_name_map: &HashMap<String, String>) -> String {
    if !line.contains("loom.bind_mem") {
        return line.to_string();
    }
    let Some((before, after)) = line.split_once(", @") else {
        return line.to_string();
    };
    let symbol_end = after
        .find(|c: char| c.is_whitespace() || c == ':' || c == ',')
        .unwrap_or(after.len());
    let symbol = &after[..symbol_end];
    let Some(prefixed) = memory_name_map.get(symbol) else {
        return line.to_string();
    };

    format!("{before}, @{prefixed}{}", &after[symbol_end..])
}

fn rewrite_mem_symbol_after_marker(
    line: &str,
    marker: &str,
    memory_name_map: &HashMap<String, String>,
) -> String {
    let Some(idx) = line.find(marker) else {
        return line.to_string();
    };

    let symbol_start = idx + marker.len();
    let after = &line[symbol_start..];
    let symbol_end = after
        .find(|c: char| c.is_whitespace() || c == ',' || c == ':')
        .unwrap_or(after.len());
    let symbol = &after[..symbol_end];
    let Some(prefixed) = memory_name_map.get(symbol) else {
        return line.to_string();
    };

    format!(
        "{}{}{}",
        &line[..symbol_start],
        prefixed,
        &after[symbol_end..]
    )
}

pub(super) fn rewrite_mlir_source(
    content: &str,
    processor_name_map: &HashMap<String, String>,
    memory_name_map: &HashMap<String, String>,
) -> String {
    let mut result = String::with_capacity(content.len());
    let mut module_rewritten = false;

    for line in content.lines() {
        let mut rewritten = line.to_string();
        if !module_rewritten {
            if let Some(module_line) = rewrite_module_symbol_line(&rewritten, processor_name_map) {
                rewritten = module_line;
                module_rewritten = true;
            }
        }
        rewritten = rewrite_bind_mem_line(&rewritten, memory_name_map);
        rewritten = rewrite_mem_symbol_after_marker(&rewritten, "src_mem_space @", memory_name_map);
        rewritten = rewrite_mem_symbol_after_marker(&rewritten, "dst_mem_space @", memory_name_map);
        result.push_str(&rewritten);
        result.push('\n');
    }

    if !content.ends_with('\n') {
        result.pop();
    }

    result
}
