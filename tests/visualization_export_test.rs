use std::fs;
use std::path::Path;

use mlar_rust::*;

#[test]
fn export_2d_mesh_visualization_yaml() {
    let architecture = mlar_rust::archs::load_arch(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/2d_mesh/processors"),
    )
    .expect("redesigned 2D mesh package should load");
    let yaml = architecture_to_visualization_yaml(&architecture)
        .expect("visualization YAML serialization should succeed");
    let document: VisualizationDocumentV1 =
        serde_yaml::from_str(&yaml).expect("visualization YAML should round-trip");

    assert_eq!(document.schema_version, VISUALIZATION_SCHEMA_VERSION);
    assert_eq!(document.architecture.name, "system");
    assert_eq!(document.scopes.len(), 2);
    assert!(document.scopes.iter().any(|scope| {
        scope.name == "x_y"
            && scope.replication_factor == Some(64)
            && scope
                .dimensions
                .iter()
                .any(|dimension| dimension.name == "x")
            && scope
                .dimensions
                .iter()
                .any(|dimension| dimension.name == "y")
    }));
    assert!(document.components.iter().any(
        |component| matches!(component, VisualizationComponent::Memory { name, .. } if name == "DRAM")
    ));
    assert_eq!(document.components.len(), 11);
    assert!(
        document
            .relationships
            .iter()
            .any(|relationship| { relationship.kind == VisualizationRelationshipKind::Read })
    );

    let output = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/2d_mesh/2d_mesh_torus.visualization.yaml");
    fs::write(output, yaml).expect("visualization YAML fixture should be writable");
}
