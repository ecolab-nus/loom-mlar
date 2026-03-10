use std::collections::{HashMap, HashSet};

use crate::arch::Sym;
use crate::mlir::MlirFuncRef;

/// Symbolic shape of one tensor operand in an operation call.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TensorShape {
    /// Tensor operand name, without `%`.
    pub tensor: String,
    /// Symbolic dimensions for this tensor (in-order), from `loom.bind`.
    pub symbols: Vec<Sym>,
}

impl TensorShape {
    pub fn new(tensor: impl Into<String>, symbols: Vec<Sym>) -> Self {
        Self {
            tensor: tensor.into(),
            symbols,
        }
    }
}

/// One operation call to a specific MLIR function symbol.
///
/// `input_shapes` and `output_shapes` encode symbolic tensor dimensions using
/// symbols declared by the function (`loom.sym`) and bindings (`loom.bind`).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Op {
    /// Function symbol name (e.g. `vec_add_f32`).
    pub name: String,
    /// Symbolic shapes for tensor inputs.
    pub input_shapes: Vec<TensorShape>,
    /// Symbolic shapes for tensor outputs.
    pub output_shapes: Vec<TensorShape>,
}

impl Op {
    pub fn new(
        name: impl Into<String>,
        input_shapes: Vec<TensorShape>,
        output_shapes: Vec<TensorShape>,
    ) -> Self {
        Self {
            name: name.into(),
            input_shapes,
            output_shapes,
        }
    }

    /// Construct an operation with no shape metadata.
    pub fn named(name: impl Into<String>) -> Self {
        Self::new(name, vec![], vec![])
    }

    /// Build an `Op` directly from parsed MLIR function metadata.
    pub fn from_mlir_func_ref(func: &MlirFuncRef) -> Self {
        let binding_by_tensor: HashMap<&str, &[Sym]> = func
            .tensor_symbol_bindings
            .iter()
            .map(|b| (b.tensor.as_str(), b.symbols.as_slice()))
            .collect();
        let output_tensors: HashSet<&str> =
            func.output_tensors.iter().map(String::as_str).collect();

        let mut input_shapes = Vec::new();
        let mut output_shapes = Vec::new();
        for tensor in &func.tensor_args {
            let shape = binding_by_tensor
                .get(tensor.as_str())
                .map(|syms| syms.to_vec())
                .unwrap_or_default();
            let op_shape = TensorShape::new(tensor.clone(), shape);

            if output_tensors.contains(tensor.as_str()) {
                output_shapes.push(op_shape);
            } else {
                input_shapes.push(op_shape);
            }
        }

        Self {
            name: func.name.clone(),
            input_shapes,
            output_shapes,
        }
    }

    /// Collect all symbols referenced by input/output shapes.
    pub fn shape_symbols(&self) -> HashSet<Sym> {
        let mut out = HashSet::new();
        for shape in self.input_shapes.iter().chain(self.output_shapes.iter()) {
            out.extend(shape.symbols.iter().cloned());
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::{Op, TensorShape};
    use crate::arch::Sym;
    use crate::mlir::MlirFuncRef;

    #[test]
    fn op_from_mlir_func_ref_splits_inputs_and_outputs() {
        let snippet = r#"
func.func @vec_add_f32(
    %L: loom.sym,
    %a: tensor<?xf32>,
    %b: tensor<?xf32>,
    %out: tensor<?xf32>
) -> tensor<?xf32> {
  loom.bind %a, (%L)
  loom.bind %b, (%L)
  loom.bind %out, (%L)
  %result = linalg.generic
    ins(%a, %b : tensor<?xf32>, tensor<?xf32>)
    outs(%out : tensor<?xf32>) { } -> tensor<?xf32>
  return %result : tensor<?xf32>
}
"#;

        let func = MlirFuncRef::from_mlir(snippet).expect("snippet should parse");
        let op = Op::from_mlir_func_ref(&func);
        assert_eq!(op.name, "vec_add_f32");
        assert_eq!(
            op.input_shapes,
            vec![
                TensorShape::new("a", vec![Sym::new("L")]),
                TensorShape::new("b", vec![Sym::new("L")]),
            ]
        );
        assert_eq!(
            op.output_shapes,
            vec![TensorShape::new("out", vec![Sym::new("L")])]
        );
    }
}
