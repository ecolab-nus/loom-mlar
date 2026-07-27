//! Loading of self-contained architecture directories.

use std::path::Path;

use crate::arch::{ArchLoadError, Architecture, SystemYaml};

/// Load `system.yaml` and its sibling `<processor>.{mlir,perf.yaml}` artifacts.
pub fn load_arch(dir: impl AsRef<Path>) -> Result<Architecture, ArchLoadError> {
    let dir = dir.as_ref();
    SystemYaml::from_file(dir.join("system.yaml"))?.build(dir)
}

#[cfg(test)]
mod tests {
    use super::load_arch;
    use crate::arch::ArchLoadError;

    #[test]
    fn missing_system_yaml_is_an_io_error() {
        let error = load_arch("tests/fixtures/architecture-does-not-exist")
            .expect_err("a missing architecture directory must not select an implicit fallback");
        assert!(matches!(error, ArchLoadError::Io { .. }));
    }
}
