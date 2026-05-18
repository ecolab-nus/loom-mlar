use crate::arch::memory::MemoryRegion;

pub(super) fn prefixed_resource_name(name: &str) -> String {
    format!("res_{name}")
}

pub(super) fn prefixed_processor_name(name: &str) -> String {
    format!("proc_{name}")
}

pub(super) fn prefixed_memory_name(name: &str) -> String {
    format!("mem_{name}")
}

pub(super) fn prefixed_arch_name(name: &str) -> String {
    format!("arch_{name}")
}

pub(super) fn prefixed_dim_name(name: &str) -> String {
    format!("dim_{name}")
}

pub(super) fn memory_region_key(region: &MemoryRegion) -> String {
    match region {
        MemoryRegion::Bank(bank) => format!(
            "bank:{:?}:{}:{:?}",
            bank.name, bank.capacity_bytes, bank.block_size
        ),
        MemoryRegion::Array {
            name,
            dims,
            sub_regions,
        } => {
            let dims = dims
                .iter()
                .map(|d| format!("{}={}", d.name.0, d.size))
                .collect::<Vec<_>>()
                .join(",");
            format!(
                "array:{:?}:[{}]:{}",
                name,
                dims,
                memory_region_key(sub_regions)
            )
        }
    }
}

pub(super) fn memory_region_resource_id(region: &MemoryRegion) -> Option<String> {
    match region {
        MemoryRegion::Bank(_) => region
            .generate_resource()
            .ok()
            .map(|resource| resource.id().as_str().to_string()),
        MemoryRegion::Array { name, .. } => name
            .as_deref()
            .or_else(|| region.name())
            .map(ToString::to_string),
    }
}
