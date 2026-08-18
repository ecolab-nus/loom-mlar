use mlar_rust::arch::ChipYaml;
use mlar_rust::{
    AffineExpr, AffineMap, Architecture, Axis, Connection, Expr, FuncPerfModel, MemoryDefinition,
    MemoryEndpoint, MlirFunc, NetworkInterface, NetworkLink, NetworkTopology, OperationModel,
    PerfScenario, ProcessorDefinition, ProcessorSelector, ProcessorTarget, Resource, Schedule,
    Scope, TimeCost, evaluate,
};
use std::path::Path;

fn memory_definition() -> MemoryDefinition {
    MemoryDefinition::new("L1", ["x", "y"], 1024, 16)
}

fn connection(input: &str, output: &str) -> Connection {
    Connection::parse(["x", "y"], [input], [output]).unwrap()
}

fn function(name: &str, latency: i64) -> OperationModel {
    OperationModel::new(
        MlirFunc::named(name),
        FuncPerfModel {
            symbols: Vec::new(),
            constraints: mlar_rust::ConstraintExpr::True,
            scenarios: vec![PerfScenario {
                constraints: mlar_rust::ConstraintExpr::True,
                time_cost: TimeCost::throughput(
                    Expr::Const(latency),
                    Expr::Const(0),
                    Expr::Const(1),
                ),
            }],
        },
    )
}

fn definition(name: &str, function_name: &str, latency: i64) -> ProcessorDefinition {
    ProcessorDefinition::new(name, "", vec![function(function_name, latency)])
}

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
    inputs: ["L1[x, y]"]
    outputs: ["L1[x, y]"]
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
fn explicit_network_and_scope_survive_canonical_construction() {
    let x = Axis::new("x", 4);
    let y = Axis::new("y", 4);
    let east = AffineMap::new(
        &[x.clone(), y.clone()],
        &[x.clone(), y.clone()],
        vec![
            AffineExpr::modulo(
                AffineExpr::add(AffineExpr::variable("x"), AffineExpr::constant(1)),
                4,
            ),
            AffineExpr::variable("y"),
        ],
    )
    .unwrap();
    let network = NetworkTopology::new("noc0", vec![Axis::new("x", 4), Axis::new("y", 4)])
        .with_resource(
            Resource::exclusive("noc0.east").indexed(vec![Axis::new("x", 4), Axis::new("y", 4)]),
        )
        .with_link(NetworkLink::new("east", east, Expr::Const(64)).with_resource("noc0.east"))
        .with_interface(NetworkInterface::new(
            "l1",
            MemoryEndpoint::parse("L1[:, :]").unwrap(),
        ));

    let architecture = Architecture::builder("mesh")
        .axis("x", 4)
        .axis("y", 4)
        .memory_definition(memory_definition())
        .place_memory("L1", ["x", "y"])
        .processor_definition(definition("lane", "op", 1))
        .connect("lane", connection("L1[x, y]", "L1[x, y]"))
        .network(network)
        .scope(
            Scope::new("mesh", ["x", "y"])
                .with_memories(["L1"])
                .with_processors(["lane"])
                .with_networks(["noc0"]),
        )
        .build()
        .expect("networked scoped architecture should build");

    assert_eq!(architecture.networks()[0].links.len(), 1);
    assert_eq!(architecture.scopes()[0].memories(), ["L1"]);
    assert_eq!(
        architecture.networks()[0].links[0]
            .map
            .apply(&[3, 2])
            .unwrap(),
        [0, 2]
    );
    let route = architecture.networks()[0]
        .shortest_route(&[3, 2], &[2, 2])
        .expect("east-only torus route");
    assert_eq!(route.len(), 3);
    assert!(
        route
            .iter()
            .all(|edge| edge.resource.as_deref() == Some("noc0.east"))
    );
    assert_eq!(route[0].resource_indices, [3, 2]);
}

#[test]
fn placed_schedule_disambiguates_duplicate_function_implementations() {
    let architecture = Architecture::builder("alternatives")
        .axis("x", 2)
        .axis("y", 1)
        .memory_definition(memory_definition())
        .place_memory("L1", ["x", "y"])
        .processor_definition(definition("fast", "op", 3))
        .processor_definition(definition("slow", "op", 9))
        .connect("fast", connection("L1[x, y]", "L1[x, y]"))
        .connect("slow", connection("L1[x, y]", "L1[x, y]"))
        .build()
        .expect("alternative implementations should be legal");

    let ambiguous = Schedule::Func {
        func: MlirFunc::named("op"),
        scenarios: None,
    };
    let error = evaluate(&ambiguous, &architecture).unwrap_err();
    assert!(error.contains("2 implementations"));
    assert!(error.contains("PlacedFunc"));

    let placed = Schedule::PlacedFunc {
        func: MlirFunc::named("op"),
        target: ProcessorTarget::select(
            "fast",
            [ProcessorSelector::Index(1), ProcessorSelector::Index(0)],
        ),
        scenarios: None,
    };
    let evaluated = evaluate(&placed, &architecture).expect("placed function should evaluate");
    let Schedule::PlacedFunc {
        scenarios: Some(scenarios),
        ..
    } = evaluated
    else {
        panic!("expected an evaluated placed function")
    };
    assert_eq!(scenarios[0].time_cost.to_expr().eval_const(), Some(3));
}

#[test]
fn parallel_schedule_uses_the_slowest_child_cost() {
    let architecture = Architecture::builder("parallel")
        .axis("x", 2)
        .axis("y", 1)
        .memory_definition(memory_definition())
        .place_memory("L1", ["x", "y"])
        .processor_definition(definition("left", "left_op", 3))
        .processor_definition(definition("right", "right_op", 9))
        .connect("left", connection("L1[x, y]", "L1[x, y]"))
        .connect("right", connection("L1[x, y]", "L1[x, y]"))
        .build()
        .unwrap();
    let schedule = Schedule::Parallel {
        schedules: vec![
            Schedule::Func {
                func: MlirFunc::named("left_op"),
                scenarios: None,
            },
            Schedule::Func {
                func: MlirFunc::named("right_op"),
                scenarios: None,
            },
        ],
        scenarios: None,
    };
    let evaluated = evaluate(&schedule, &architecture).unwrap();
    let Schedule::Parallel {
        scenarios: Some(scenarios),
        ..
    } = evaluated
    else {
        panic!("expected evaluated parallel schedule")
    };
    assert_eq!(scenarios[0].time_cost.to_expr().eval_const(), Some(9));
}

#[test]
fn raw_mlir_is_an_alternate_processor_frontend() {
    let source = r#"
module @lane {
  func.func @add(%src: memref<?xf16>, %dst: memref<?xf16>) {
    %L = loom.sym @L : index
    loom.bind_shape %src, [%L] : memref<?xf16>
    loom.bind_shape %dst, [%L] : memref<?xf16>
    loom.bind_mem %src, @L1 : memref<?xf16>
    loom.bind_mem %dst, @L1 : memref<?xf16>
    linalg.copy ins(%src : memref<?xf16>) outs(%dst : memref<?xf16>)
    return
  }
}
"#;
    let perf = FuncPerfModel::builder()
        .symbols(["L"])
        .simple_time_cost(Expr::Const(1), Expr::sym("L"), Expr::Const(32))
        .build();
    let definition = ProcessorDefinition::from_mlir_source("lane", source, [("add", perf)])
        .expect("raw MLIR should parse");
    assert_eq!(definition.operations()[0].func.name, "add");
    assert!(matches!(
        definition.source_format(),
        mlar_rust::ProcessorSourceFormat::Mlir
    ));

    let architecture = Architecture::builder("raw_mlir")
        .axis("x", 1)
        .axis("y", 1)
        .memory_definition(memory_definition())
        .place_memory("L1", ["x", "y"])
        .processor_definition(definition.with_type(mlar_rust::ProcessorType::Compute))
        .connect("lane", connection("L1[x, y]", "L1[x, y]"))
        .build()
        .expect("raw MLIR architecture should build");
    let exported = mlar_rust::architecture_to_mlir(&architecture)
        .expect("raw MLIR architecture should export");
    assert!(exported.contains("module @proc_lane"));
    assert!(exported.contains("loom.bind_mem %src, @mem_L1"));
}
