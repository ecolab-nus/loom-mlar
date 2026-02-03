#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "MlarDialect.h.inc"
// Generated type declarations
#define GET_TYPEDEF_CLASSES
#include "MlarTypes.h.inc"
// Generated type definitions (TypeID, printers/parsers)
#define GET_TYPEDEF_CLASSES
#include "MlarTypes.cpp.inc"

using namespace mlir;
using namespace loom::mlar;

#include "MlarDialect.cpp.inc"
// Bring in op class declarations for registration below.
#define GET_OP_CLASSES
#include "MlarOps.h.inc"

void MlarDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "MlarOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "MlarTypes.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "MlarOps.cpp.inc"

//===----------------------------------------------------------------------===//
// FuOp Custom Assembly Format
//===----------------------------------------------------------------------===//

/// Parse FuOp with custom syntax:
///   mlar.fu @func_name <%x, %y>
///   or: mlar.fu @func_name  (no dims = single instance)
ParseResult FuOp::parse(OpAsmParser &parser, OperationState &result) {
  Builder &builder = parser.getBuilder();
  
  // Parse function reference
  FlatSymbolRefAttr funcRef;
  if (parser.parseAttribute(funcRef, "func_ref", result.attributes))
    return failure();

  // Parse optional <...> clause for dimensions
  SmallVector<OpAsmParser::UnresolvedOperand> dimsOperands;
  if (succeeded(parser.parseOptionalLess())) {
    if (parser.parseOperandList(dimsOperands))
      return failure();
    
    if (parser.parseGreater())
      return failure();
  }

  // Parse optional attribute dictionary
  if (parser.parseOptionalAttrDict(result.attributes).failed())
    return failure();

  // Resolve dims operands (all should be index type)
  SmallVector<Type> dimsTypes(dimsOperands.size(), builder.getIndexType());
  if (!dimsOperands.empty() &&
      parser.resolveOperands(dimsOperands, dimsTypes, parser.getNameLoc(),
                             result.operands))
    return failure();

  // Set result type
  result.addTypes(FunctionalUnitHandleType::get(builder.getContext()));

  return success();
}

/// Print FuOp with custom syntax
void FuOp::print(OpAsmPrinter &p) {
  p << " ";
  p.printAttributeWithoutType(getFuncRefAttr());
  
  if (!getDims().empty()) {
    p << " <";
    p.printOperands(getDims());
    p << ">";
  }
  
  // Elide func_ref from attribute dict since it's printed separately
  SmallVector<StringRef> elidedAttrs = {"func_ref"};
  p.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

//===----------------------------------------------------------------------===//
// LaneOp Custom Assembly Format
//===----------------------------------------------------------------------===//

/// Parse LaneOp with custom syntax:
///   mlar.lane @func_name <%x, %y>
///   or: mlar.lane @func_name  (no dims = single instance)
ParseResult LaneOp::parse(OpAsmParser &parser, OperationState &result) {
  Builder &builder = parser.getBuilder();
  
  // Parse function reference
  FlatSymbolRefAttr funcRef;
  if (parser.parseAttribute(funcRef, "func_ref", result.attributes))
    return failure();

  // Parse optional <...> clause for dimensions
  SmallVector<OpAsmParser::UnresolvedOperand> dimsOperands;
  if (succeeded(parser.parseOptionalLess())) {
    if (parser.parseOperandList(dimsOperands))
      return failure();
    
    if (parser.parseGreater())
      return failure();
  }

  // Parse optional attribute dictionary
  if (parser.parseOptionalAttrDict(result.attributes).failed())
    return failure();

  // Resolve dims operands (all should be index type)
  SmallVector<Type> dimsTypes(dimsOperands.size(), builder.getIndexType());
  if (!dimsOperands.empty() &&
      parser.resolveOperands(dimsOperands, dimsTypes, parser.getNameLoc(),
                             result.operands))
    return failure();

  // Set result type
  result.addTypes(FunctionalUnitHandleType::get(builder.getContext()));

  return success();
}

/// Print LaneOp with custom syntax
void LaneOp::print(OpAsmPrinter &p) {
  p << " ";
  p.printAttributeWithoutType(getFuncRefAttr());
  
  if (!getDims().empty()) {
    p << " <";
    p.printOperands(getDims());
    p << ">";
  }
  
  // Elide func_ref from attribute dict since it's printed separately
  SmallVector<StringRef> elidedAttrs = {"func_ref"};
  p.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

//===----------------------------------------------------------------------===//
// CoreOp Custom Assembly Format
//===----------------------------------------------------------------------===//

/// Parse CoreOp with custom syntax:
///   mlar.core "label" {scaleout=(%x, %y), scalein=(%mat_unit, %vec_unit, [8,1])}

//===----------------------------------------------------------------------===//
// MemoryOp Custom Assembly Format
//===----------------------------------------------------------------------===//

/// Parse MemoryOp with custom syntax:
///   mlar.memory "L1" nb_blocks block_size <%x, %y>
ParseResult MemoryOp::parse(OpAsmParser &parser, OperationState &result) {
  Builder &builder = parser.getBuilder();
  MLIRContext *context = builder.getContext();
  
  // Parse sym_name attribute
  StringAttr symName;
  if (parser.parseAttribute(symName, "sym_name", result.attributes))
    return failure();

  // Parse nb_blocks (integer literal)
  int64_t nbBlocks;
  if (parser.parseInteger(nbBlocks))
    return failure();
  result.addAttribute("nb_blocks", builder.getI64IntegerAttr(nbBlocks));

  // Parse port_width (integer literal)
  int64_t portWidth;
  if (parser.parseInteger(portWidth))
    return failure();
  result.addAttribute("port_width", builder.getI64IntegerAttr(portWidth));

  // Parse <dims...> clause
  SmallVector<OpAsmParser::UnresolvedOperand> dimsOperands;
  if (parser.parseLess())
    return failure();
  
  if (parser.parseOperandList(dimsOperands))
    return failure();
  
  if (parser.parseGreater())
    return failure();

  // Parse result type annotation
  Type resultType;
  if (parser.parseColon() || parser.parseType(resultType))
    return failure();

  // Parse optional attribute dictionary
  if (parser.parseOptionalAttrDict(result.attributes).failed())
    return failure();

  // Resolve dims operands (all should be index type)
  SmallVector<Type> dimsTypes(dimsOperands.size(), builder.getIndexType());
  if (parser.resolveOperands(dimsOperands, dimsTypes, parser.getNameLoc(),
                             result.operands))
    return failure();

  // Set result type from parsed type
  result.addTypes(resultType);

  return success();
}

/// Print MemoryOp with custom syntax
void MemoryOp::print(OpAsmPrinter &p) {
  p << " ";
  p.printAttributeWithoutType(getLabelAttr());
  
  p << " " << getNbBlocks();
  p << " " << getPortWidth();
  
  if (!getDims().empty()) {
    p << " <";
    p.printOperands(getDims());
    p << ">";
  }
  
  // Print result type
  p << " : ";
  p.printType(getMemref().getType());
  
  // Elide sym_name, nb_blocks, port_width from attribute dict since they're printed separately
  SmallVector<StringRef> elidedAttrs = {"sym_name", "nb_blocks", "port_width"};
  p.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

//===----------------------------------------------------------------------===//
// MuxOp Custom Assembly Format
//===----------------------------------------------------------------------===//

/// Parse MuxOp with custom syntax:
///   mlar.mux %cores, %memories, {map = affine_map<(d0, d1) -> (d0, d1)>}
///   or: mlar.mux %cores: !mlar.compute, %memories: !mlar.memory, %x, %y {map = ...}

//===----------------------------------------------------------------------===//
// InterconnectsOp Custom Assembly Format
//===----------------------------------------------------------------------===//

/// Parse InterconnectsOp with new syntax:
///   mlar.interconnects @func_ref <%x, %y> {map = affine_map<...>}
ParseResult InterconnectsOp::parse(OpAsmParser &parser, OperationState &result) {
  Builder &builder = parser.getBuilder();
  MLIRContext *context = builder.getContext();

  // Parse function reference (@func_name)
  FlatSymbolRefAttr funcRef;
  if (parser.parseAttribute(funcRef, "func_ref", result.attributes))
    return failure();

  // Parse spatial dimensions in angle brackets: <%x, %y>
  SmallVector<OpAsmParser::UnresolvedOperand> dims;
  if (parser.parseLess() || 
      parser.parseOperandList(dims) || 
      parser.parseGreater())
    return failure();

  // Parse attributes dictionary (must contain 'map')
  if (parser.parseOptionalAttrDict(result.attributes).failed())
    return failure();

  // Resolve dimension operands (all should be index type)
  SmallVector<Type> dimTypes(dims.size(), builder.getIndexType());
  if (parser.resolveOperands(dims, dimTypes, parser.getNameLoc(), result.operands))
    return failure();

  // Set result type
  result.addTypes(InterconnectHandleType::get(context));

  return success();
}

/// Print InterconnectsOp with new syntax:
///   %result = mlar.interconnects @func_ref <%x, %y> {map = affine_map<...>}
void InterconnectsOp::print(OpAsmPrinter &p) {
  p << " ";
  
  // Print function reference
  p.printAttributeWithoutType(getFuncRefAttr());
  
  // Print spatial dimensions in angle brackets
  p << " <";
  p.printOperands(getDims());
  p << ">";
  
  // Print attributes (map, etc.)
  p << " ";
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"func_ref"});
}

