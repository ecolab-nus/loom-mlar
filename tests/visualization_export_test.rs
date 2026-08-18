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
    for resource_name in ["matrix_lane", "vector_lane", "noc0", "noc1"] {
        let matching = document
            .components
            .iter()
            .filter(|component| {
                matches!(
                    component,
                    VisualizationComponent::Resource { name, .. } if name == resource_name
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(
            matching.len(),
            1,
            "resource '{resource_name}' should be exported once"
        );
    }
    let scope_of = |name: &str| {
        document
            .scopes
            .iter()
            .find(|scope| scope.name == name)
            .unwrap_or_else(|| panic!("scope '{name}' should be exported"))
            .id
            .as_str()
    };
    for (resource_name, scope_name) in [
        ("matrix_lane", "x_y"),
        ("vector_lane", "x_y"),
        ("noc0", "system"),
        ("noc1", "system"),
    ] {
        let component = document
            .components
            .iter()
            .find(|component| {
                matches!(
                    component,
                    VisualizationComponent::Resource { name, .. } if name == resource_name
                )
            })
            .expect("resource should be exported");
        assert_eq!(
            component.scope(),
            scope_of(scope_name),
            "resource '{resource_name}' should belong to scope '{scope_name}'"
        );
    }
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
