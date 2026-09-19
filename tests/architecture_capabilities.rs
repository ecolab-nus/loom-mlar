use mlar_rust::{
    AffineExpr, AffineMap, Architecture, Axis, Connection, Expr, FuncPerfModel, MemoryDefinition,
    MlirFunc, NetworkInterface, NetworkLink, NetworkTopology, OperationModel, PerfScenario,
    ProcessorDefinition, ProcessorSelector, ProcessorTarget, Resource, Schedule, Scope, TimeCost,
    evaluate,
};

fn memory_definition() -> MemoryDefinition {
    MemoryDefinition::new("L1", 1024, 16)
}

fn connection(input: &str, output: &str) -> Connection {
    Connection::new(["x", "y"])
        .input("input", endpoint(input))
        .output("result", endpoint(output))
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
        .with_interface(NetworkInterface::new("l1", endpoint("L1[:, :]")));

    let architecture = Architecture::builder("mesh")
        .axis("x", 4)
        .axis("y", 4)
        .memory_definition(memory_definition())
        .place_memory("L1", mlar_rust::MemoryDomain::L1, ["x", "y"])
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
        .place_memory("L1", mlar_rust::MemoryDomain::L1, ["x", "y"])
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
        .place_memory("L1", mlar_rust::MemoryDomain::L1, ["x", "y"])
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
    loom.bind_mem %src, @input : memref<?xf16>
    loom.bind_mem %dst, @result : memref<?xf16>
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

    let architecture = Architecture::builder("raw_mlir")
        .axis("x", 1)
        .axis("y", 1)
        .memory_definition(memory_definition())
        .place_memory("L1", mlar_rust::MemoryDomain::L1, ["x", "y"])
        .processor_definition(definition.with_type(mlar_rust::ProcessorType::Compute))
        .connect(
            "lane",
            Connection::new(["x", "y"])
                .input("input", "L1")
                .output("result", "L1"),
        )
        .build()
        .expect("raw MLIR architecture should build");
    let lane = architecture.processor_array("lane").unwrap();
    assert_eq!(lane.connection().inputs[0].name, "input");
    assert_eq!(lane.connection().inputs[0].endpoint.indices.len(), 2);
    let exported = mlar_rust::architecture_to_mlir(&architecture)
        .expect("raw MLIR architecture should export");
    assert!(exported.contains("module @proc_lane"));
    assert!(exported.contains("loom.bind_mem %src, @mem_L1"));
}

#[test]
fn raw_mlir_regions_must_match_named_connection_ports() {
    let source = r#"
module @lane {
  func.func @copy(%src: memref<?xf16>, %dst: memref<?xf16>) {
    %L = loom.sym @L : index
    loom.bind_shape %src, [%L] : memref<?xf16>
    loom.bind_shape %dst, [%L] : memref<?xf16>
    loom.bind_mem %src, @DRAM : memref<?xf16>
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
    let definition = ProcessorDefinition::from_mlir_source("lane", source, [("copy", perf)])
        .unwrap()
        .with_type(mlar_rust::ProcessorType::DataMover);
    let error = Architecture::builder("named_ports")
        .memory_definition(memory_definition())
        .place_memory("L1", mlar_rust::MemoryDomain::L1, Vec::<String>::new())
        .processor_definition(definition)
        .connect(
            "lane",
            Connection::new(Vec::<String>::new())
                .input("src", "L1")
                .output("dst", "L1"),
        )
        .build()
        .unwrap_err();
    assert!(error.to_string().contains("@DRAM"));
    assert!(error.to_string().contains("not declared by the connection"));
}

#[test]
fn one_port_name_cannot_hide_two_memory_endpoints() {
    let error = Architecture::builder("ambiguous_port")
        .memory_definition(MemoryDefinition::new("A", 1024, 16))
        .memory_definition(MemoryDefinition::new("B", 1024, 16))
        .place_memory("A", mlar_rust::MemoryDomain::L1, Vec::<String>::new())
        .place_memory("B", mlar_rust::MemoryDomain::L1, Vec::<String>::new())
        .processor_definition(
            ProcessorDefinition::new("lane", "", Vec::new())
                .with_type(mlar_rust::ProcessorType::Compute),
        )
        .connect(
            "lane",
            Connection::new(Vec::<String>::new())
                .input("data", "A")
                .output("data", "B"),
        )
        .build()
        .unwrap_err();
    assert!(error.to_string().contains("different memory endpoints"));
}

fn endpoint(text: &str) -> mlar_rust::MemoryEndpoint {
    let (name, selectors) = text.split_once('[').unwrap();
    mlar_rust::MemoryEndpoint::new(
        name,
        selectors
            .trim_end_matches(']')
            .split(',')
            .map(|selector| {
                if selector.trim() == ":" {
                    mlar_rust::arch::EndpointIndex::All
                } else {
                    mlar_rust::arch::EndpointIndex::Expression(
                        mlar_rust::AffineExpr::parse(selector.trim()).unwrap(),
                    )
                }
            })
            .collect(),
    )
}
