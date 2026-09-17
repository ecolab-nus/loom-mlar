use serde::Serialize;
use std::collections::BTreeMap;
use std::path::Path;

use crate::builder::FunctionProvider;
use crate::native::NativeSourceIndex;
use crate::{ArchLoadError, ChipYaml, ProcessorYaml};

#[derive(Serialize)]
struct Manifest {
    processors: BTreeMap<String, BTreeMap<String, SourceRecord>>,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
enum SourceRecord {
    Template { name: String },
    Native { path: String },
}

pub fn emit_processor_sources(
    package_dir: impl AsRef<Path>,
    output_dir: impl AsRef<Path>,
) -> Result<(), ArchLoadError> {
    let package_dir = package_dir.as_ref();
    let output_dir = output_dir.as_ref();
    reject_output_inside_package(package_dir, output_dir)?;
    if output_dir.exists() {
        return Err(ArchLoadError::Invalid(format!(
            "processor emission output '{}' already exists",
            output_dir.display()
        )));
    }

    let chip = ChipYaml::from_file(package_dir.join("chip.yaml"))?;
    let architecture = chip.build(package_dir)?;
    let native_sources = NativeSourceIndex::load(package_dir).map_err(ArchLoadError::Invalid)?;
    let mut authored = BTreeMap::new();
    for path in chip.processor_definition_paths(package_dir) {
        let yaml = ProcessorYaml::from_file(&path)?;
        let definition = yaml.build_definition_with_index(&path, &native_sources)?;
        let sources = definition
            .providers
            .as_ref()
            .expect("YAML definitions retain providers")
            .iter()
            .map(|(function, provider)| {
                let source = match provider {
                    FunctionProvider::Template(spec) => SourceRecord::Template {
                        name: spec.source.clone(),
                    },
                    FunctionProvider::Native(native) => SourceRecord::Native {
                        path: native
                            .path
                            .strip_prefix(package_dir)
                            .unwrap_or(&native.path)
                            .display()
                            .to_string(),
                    },
                };
                (function.clone(), source)
            })
            .collect::<BTreeMap<_, _>>();
        authored.insert(definition.name().to_string(), sources);
    }

    std::fs::create_dir(output_dir).map_err(|source| ArchLoadError::Io {
        path: output_dir.to_path_buf(),
        source,
    })?;
    let mut manifest = Manifest {
        processors: BTreeMap::new(),
    };
    for definition in architecture.processor_definitions() {
        validate_filename(definition.name())?;
        let path = output_dir.join(format!("{}.mlir", definition.name()));
        std::fs::write(&path, definition.source()).map_err(|source| ArchLoadError::Io {
            path: path.clone(),
            source,
        })?;
        let base = authored
            .keys()
            .filter(|name| {
                definition.name() == name.as_str()
                    || definition.name().starts_with(&format!("{name}__"))
            })
            .max_by_key(|name| name.len())
            .ok_or_else(|| {
                ArchLoadError::Invalid(format!(
                    "no authored source record for resolved processor '{}'",
                    definition.name()
                ))
            })?;
        manifest
            .processors
            .insert(definition.name().to_string(), authored[base].clone());
    }
    let manifest_path = output_dir.join("manifest.yaml");
    let yaml = serde_yaml::to_string(&manifest)
        .map_err(|error| ArchLoadError::Invalid(error.to_string()))?;
    std::fs::write(&manifest_path, yaml).map_err(|source| ArchLoadError::Io {
        path: manifest_path,
        source,
    })?;
    Ok(())
}

fn reject_output_inside_package(package: &Path, output: &Path) -> Result<(), ArchLoadError> {
    let package = std::fs::canonicalize(package).map_err(|source| ArchLoadError::Io {
        path: package.to_path_buf(),
        source,
    })?;
    let parent = output.parent().unwrap_or_else(|| Path::new("."));
    let parent = if parent.as_os_str().is_empty() {
        Path::new(".")
    } else {
        parent
    };
    let parent = std::fs::canonicalize(parent).map_err(|source| ArchLoadError::Io {
        path: parent.to_path_buf(),
        source,
    })?;
    let output = parent.join(output.file_name().unwrap_or_default());
    if output.starts_with(&package) {
        return Err(ArchLoadError::Invalid(format!(
            "processor emission output '{}' must be outside the source package",
            output.display()
        )));
    }
    Ok(())
}

fn validate_filename(name: &str) -> Result<(), ArchLoadError> {
    if name.is_empty()
        || !name
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || matches!(character, '_' | '-'))
    {
        return Err(ArchLoadError::Invalid(format!(
            "processor name '{name}' cannot be used as an emitted filename"
        )));
    }
    Ok(())
}
