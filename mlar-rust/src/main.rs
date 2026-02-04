use mlar_rust::*;
use lane::{MatMulLane, VecLane};
use interconnect::AffineExpr;

fn main() {
    println!("=== MLAR Rust Prototype ===\n");

    // Create a 2D mesh architecture similar to 2d_mesh.mlir
    let arch = create_2d_mesh_architecture();

    println!("Architecture: {}", arch.name);
    println!("Grid dimensions:");
    for dim in &arch.dimensions {
        println!("  {}: {} units", dim.name, dim.size);
    }
    println!("Total processing elements: {}", 
             arch.total_processing_elements()
                 .map(|n| n.to_string())
                 .unwrap_or_else(|| "symbolic".to_string()));
    
    println!("\nFunctional Units: {}", arch.functional_units.len());
    for fu in &arch.functional_units {
        println!("  {} - Latency: {} cycles", fu.name, fu.latency);
    }

    println!("\nLanes: {}", arch.lanes.len());
    for lane in &arch.lanes {
        println!("  {}", lane.name);
    }

    println!("\nMemories: {}", arch.memories.len());
    for mem in &arch.memories {
        println!("  {} - Capacity: {} bytes, Bandwidth: {} bytes/cycle",
                 mem.name, mem.capacity, mem.bandwidth);
    }

    println!("\nInterconnects: {}", arch.interconnects.len());
    for ic in &arch.interconnects {
        println!("  {} - Bandwidth: {} bytes/cycle", ic.name, ic.bandwidth);
    }

    // Test affine maps
    println!("\n=== Testing Affine Maps ===");
    test_affine_maps(&arch);

    // Test lane preconditions
    println!("\n=== Testing Lane Preconditions ===");
    test_lane_preconditions();

    // Test symbolic sizes
    println!("\n=== Testing Symbolic Sizes ===");
    test_symbolic_sizes();
}

fn create_2d_mesh_architecture() -> Architecture {
    // Define grid dimensions
    let dim_x = Dimension::new("x", 8);
    let dim_y = Dimension::new("y", 8);
    let dim_d = Dimension::new("d", 4);

    // Create functional units
    let mat_fu = FunctionalUnit::builder("matmul_32x32")
        .input(MemRef::new_static(vec![32, 32], "f32"))
        .input(MemRef::new_static(vec![32, 32], "f32"))
        .output(MemRef::new_static(vec![32, 32], "f32"))
        .latency(8)
        .grid(vec![dim_x.clone(), dim_y.clone()])
        .build();

    let vec_fu = FunctionalUnit::builder("vec_add_32")
        .input(MemRef::new_static(vec![32], "f32"))
        .input(MemRef::new_static(vec![32], "f32"))
        .output(MemRef::new_static(vec![32], "f32"))
        .latency(1)
        .grid(vec![dim_x.clone(), dim_y.clone()])
        .build();

    // Create lanes
    let mat_lane = Lane::new(
        "matmul_lane",
        Box::new(MatMulLane),
        vec![dim_x.clone(), dim_y.clone()],
    );

    let vec_lane = Lane::new(
        "vec_lane",
        Box::new(VecLane),
        vec![dim_x.clone(), dim_y.clone()],
    );

    // Create memories
    let l1 = Memory::builder("L1")
        .capacity(65536)
        .bandwidth(16)
        .grid(vec![dim_x.clone(), dim_y.clone()])
        .build();

    let dram = Memory::builder("DRAM")
        .capacity(34359738368) // 32 GB
        .bandwidth(288)
        .grid(vec![dim_d.clone()])
        .build();

    // Create interconnects with affine maps
    // Horizontal NoC: (d0, d1) -> ((d0 + 1) mod 8, d1)
    let noc_h_map = AffineMap::new(
        2,
        vec![
            AffineExpr::modulo(
                AffineExpr::add(AffineExpr::dim(0), AffineExpr::constant(1)),
                AffineExpr::constant(8),
            ),
            AffineExpr::dim(1),
        ],
    );

    let noc_h = Interconnect::builder("horizontal_noc")
        .grid(vec![dim_x.clone(), dim_y.clone()])
        .affine_map(noc_h_map)
        .bandwidth(32)
        .build();

    // Vertical NoC: (d0, d1) -> (d0, (d1 + 1) mod 8)
    let noc_v_map = AffineMap::new(
        2,
        vec![
            AffineExpr::dim(0),
            AffineExpr::modulo(
                AffineExpr::add(AffineExpr::dim(1), AffineExpr::constant(1)),
                AffineExpr::constant(8),
            ),
        ],
    );

    let noc_v = Interconnect::builder("vertical_noc")
        .grid(vec![dim_x.clone(), dim_y.clone()])
        .affine_map(noc_v_map)
        .bandwidth(32)
        .build();

    // L1 to DRAM: (d0, d1) -> (d0 ceildiv 4 + 2 * (d1 ceildiv 4))
    let to_dram_map = AffineMap::new(
        2,
        vec![AffineExpr::add(
            AffineExpr::ceildiv(AffineExpr::dim(0), AffineExpr::constant(4)),
            AffineExpr::mul(
                AffineExpr::constant(2),
                AffineExpr::ceildiv(AffineExpr::dim(1), AffineExpr::constant(4)),
            ),
        )],
    );

    let to_dram = Interconnect::builder("to_dram")
        .grid(vec![dim_x.clone(), dim_y.clone()])
        .affine_map(to_dram_map)
        .bandwidth(64)
        .build();

    // Build the complete architecture
    Architecture::builder("2D Mesh")
        .dimension(dim_x)
        .dimension(dim_y)
        .dimension(dim_d)
        .functional_unit(mat_fu)
        .functional_unit(vec_fu)
        .lane(mat_lane)
        .lane(vec_lane)
        .memory(l1)
        .memory(dram)
        .interconnect(noc_h)
        .interconnect(noc_v)
        .interconnect(to_dram)
        .build()
}

fn test_affine_maps(arch: &Architecture) {
    // Test horizontal NoC ring
    let noc_h = &arch.interconnects[0];
    println!("Horizontal NoC connections:");
    for x in 0..4 {
        for y in 0..4 {
            let target = noc_h.get_target(&[x, y]);
            println!("  ({}, {}) -> ({}, {})", x, y, target[0], target[1]);
        }
    }

    // Test DRAM mapping
    let to_dram = &arch.interconnects[2];
    println!("\nL1 to DRAM mapping:");
    for x in 0..8 {
        for y in 0..8 {
            let dram_idx = to_dram.get_target(&[x, y]);
            println!("  Core ({}, {}) -> DRAM {}", x, y, dram_idx[0]);
        }
    }
}

fn test_lane_preconditions() {
    let mat_lane = Lane::new("test_matmul_lane", Box::new(MatMulLane), vec![]);
    
    // Test valid case
    let result = mat_lane.compute_latency(&[512, 512, 512], &[]);
    match result {
        Ok(latency) => println!("MatMul(512x512x512) latency: {} cycles", latency),
        Err(e) => println!("Error: {}", e),
    }

    // Test invalid case (too small)
    let result = mat_lane.compute_latency(&[128, 128, 128], &[]);
    match result {
        Ok(latency) => println!("MatMul(128x128x128) latency: {} cycles", latency),
        Err(e) => println!("Expected error: {}", e),
    }

    let vec_lane = Lane::new("test_vec_lane", Box::new(VecLane), vec![]);
    
    // Test valid case
    let result = vec_lane.compute_latency(&[2048], &[]);
    match result {
        Ok(latency) => println!("VecAdd(2048) latency: {} cycles", latency),
        Err(e) => println!("Error: {}", e),
    }

    // Test invalid case (too small)
    let result = vec_lane.compute_latency(&[512], &[]);
    match result {
        Ok(latency) => println!("VecAdd(512) latency: {} cycles", latency),
        Err(e) => println!("Expected error: {}", e),
    }
}

fn test_symbolic_sizes() {
    use crate::primitives::Size;
    
    // Create dimensions with symbolic sizes
    let dim_x = Dimension::new_symbolic("x", "N");
    let dim_y = Dimension::new_symbolic("y", "M");
    let dim_z = Dimension::new("z", 16); // Mix of symbolic and concrete
    
    println!("Dimensions with symbolic sizes:");
    println!("  {}: {}", dim_x.name, dim_x.size);
    println!("  {}: {}", dim_y.name, dim_y.size);
    println!("  {}: {}", dim_z.name, dim_z.size);
    
    // Create architecture with symbolic dimensions
    let arch = Architecture::builder("Symbolic Mesh")
        .dimension(dim_x.clone())
        .dimension(dim_y.clone())
        .dimension(dim_z)
        .build();
    
    println!("\nArchitecture: {}", arch.name);
    println!("Total processing elements: {}", 
             arch.total_processing_elements()
                 .map(|n| n.to_string())
                 .unwrap_or_else(|| "symbolic (contains N, M)".to_string()));
    
    // Create MemRefs with symbolic shapes
    let symbolic_memref = MemRef::new_static_sizes(
        vec![Size::symbolic("N"), Size::symbolic("M")],
        "f32"
    );
    
    let mixed_memref = MemRef::new_static_sizes(
        vec![Size::concrete(32), Size::symbolic("K"), Size::concrete(32)],
        "f32"
    );
    
    println!("\nMemRef with symbolic shape:");
    match &symbolic_memref.shape {
        crate::primitives::Shape::Static(sizes) => {
            print!("  memref<");
            for (i, size) in sizes.iter().enumerate() {
                if i > 0 { print!("x"); }
                print!("{}", size);
            }
            println!("xf32>");
        }
        _ => {}
    }
    
    println!("\nMemRef with mixed concrete/symbolic shape:");
    match &mixed_memref.shape {
        crate::primitives::Shape::Static(sizes) => {
            print!("  memref<");
            for (i, size) in sizes.iter().enumerate() {
                if i > 0 { print!("x"); }
                print!("{}", size);
            }
            println!("xf32>");
        }
        _ => {}
    }
    
    // Demonstrate Size checking
    println!("\nSize type checking:");
    let concrete = Size::concrete(64);
    let symbolic = Size::symbolic("TILE_SIZE");
    
    println!("  {} is concrete: {}", concrete, concrete.is_concrete());
    println!("  {} is symbolic: {}", symbolic, symbolic.is_symbolic());
    println!("  {} concrete value: {:?}", concrete, concrete.as_concrete());
    println!("  {} symbolic name: {:?}", symbolic, symbolic.as_symbolic());
}
