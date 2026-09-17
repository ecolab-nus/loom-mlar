use mlar_frontend::ChipYaml;
use std::path::Path;
#[test]
fn declarative_package_supports_parameters_networks_and_scopes() {
    let source = r#"
name: declarative_mesh
parameters: [X, Y]
dimensions:
  channel: 2
  lx: 2
  ly: 2
  x: X
  y: "Y * 2"
memories:
  DRAM: [channel]
  L1: [x, y]
  L2: [lx, ly]
networks:
  - name: torus
    dimensions: [x, y]
    resources:
      - name: east_links
    links:
      - name: east
        map: "[x, y] -> [x, y]: ((x + 1) mod X, y)"
        bandwidth: "Y * 32"
        resource: east_links
    interfaces:
      - name: l1
        endpoint: "L1[:, :]"
scopes:
  - name: mesh
    dimensions: [x, y]
    memories: [L1]
    processors: [matrix_lane]
    networks: [torus]
processors:
  matrix_lane:
    definition: matrix_lane.yaml
    domain: [x, y]
    inputs: {lhs: "L1[x, y]", rhs: "L1[x, y]"}
    outputs: {result: "L1[x, y]"}
"#;
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/indexed-affine");
    let chip = ChipYaml::from_yaml_str(source).expect("parameterized YAML syntax");
    let architecture = chip
        .build_with_bindings(&fixture, [("X", 4), ("Y", 2)])
        .expect("parameterized package should build");

    assert_eq!(architecture.axis("y").unwrap().extent(), 4);
    assert_eq!(
        architecture.networks()[0].links[0]
            .map
            .apply(&[3, 1])
            .unwrap(),
        [0, 1]
    );
    assert_eq!(
        architecture.networks()[0].links[0].bandwidth.eval_const(),
        Some(64)
    );
    assert_eq!(architecture.scopes()[0].name(), "mesh");
    let mlir = mlar_rust::architecture_to_mlir(&architecture)
        .expect("explicit scope should drive valid ADL lowering");
    assert!(mlir.contains("adl.arch.scale \"arch_mesh\""));
}

#[test]
fn shared_resources_resolve_declared_dimensions_and_parameter_extents() {
    let chip = ChipYaml::from_yaml_str(
        r#"
name: resources
parameters: [X]
dimensions: {x: X, y: 3}
resources:
- name: stage_port
  dimensions: [y, x]
- name: global_port
- name: slots
  capacity: 4
  dimensions: [x]
"#,
    )
    .unwrap();
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/typed-memory");
    let architecture = chip.build_with_bindings(&fixture, [("X", 2)]).unwrap();
    let resources = architecture.resources();
    assert_eq!(
        resources[0].axes(),
        [mlar_rust::Axis::new("y", 3), mlar_rust::Axis::new("x", 2)]
    );
    assert!(resources[1].axes().is_empty());
    assert_eq!(resources[2].capacity(), Some(4));
    assert_eq!(resources[2].axes(), [mlar_rust::Axis::new("x", 2)]);
}

#[test]
fn shared_resources_reject_unknown_dimensions() {
    let chip = ChipYaml::from_yaml_str(
        "name: resources\nresources:\n- name: stage_port\n  dimensions: [missing]\n",
    )
    .unwrap();
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/typed-memory");
    let error = chip.build(&fixture).unwrap_err().to_string();
    assert!(
        error.contains("resource 'stage_port' uses unknown dimension 'missing'"),
        "{error}"
    );
}
