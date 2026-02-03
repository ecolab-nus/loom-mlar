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

/// Parse InterconnectsOp with custom syntax:
///   mlar.interconnects "horizontal_links" %memories, %memories, {map = ..., bandwidth = 128}
///   or: mlar.interconnects %memories: !mlar.memory, %drams : !mlar.memory, {map = ...}
/// 
/// This parser uses a flexible function-object based approach to handle various
/// format combinations (with/without sym_name, type annotations, indices, etc.)
ParseResult InterconnectsOp::parse(OpAsmParser &parser, OperationState &result) {
  Builder &builder = parser.getBuilder();
  MLIRContext *context = builder.getContext();

  // Parse symbol name (required for Symbol trait, but allow optional for backward compatibility)
  StringAttr symName;
  std::string symNameStr;
  if (succeeded(parser.parseOptionalString(&symNameStr))) {
    symName = builder.getStringAttr(symNameStr);
  } else {
    // Generate a default name if not provided (for backward compatibility)
    // Use a simple default name - if there are conflicts, MLIR's Symbol system will report an error
    // In practice, users should provide explicit names for interconnects
    symName = builder.getStringAttr("interconnect");
  }
  result.addAttribute("sym_name", symName);

  // Helper function to parse an operand with optional type annotation
  // Returns the operand and whether a type was provided
  auto parseOperandWithType = [&](OpAsmParser::UnresolvedOperand &operand,
                                  Type &type, bool &hasType) -> ParseResult {
    if (parser.parseOperand(operand))
      return failure();
    
    hasType = false;
    if (succeeded(parser.parseOptionalColon())) {
      if (parser.parseType(type).failed())
        return failure();
      hasType = true;
    }
    return success();
  };

  // Helper function to validate that a type is a valid mlar handle type
  auto isValidHandleType = [](Type type) -> bool {
    return llvm::isa<MemoryHandleType>(type);
  };

  // Parse source operand with optional type
  OpAsmParser::UnresolvedOperand source;
  Type sourceType;
  bool hasSourceType = false;
  if (parseOperandWithType(source, sourceType, hasSourceType).failed())
    return failure();

  // Validate source type if provided
  if (hasSourceType && !isValidHandleType(sourceType)) {
    return parser.emitError(parser.getNameLoc(),
                           "source type must be a mlar handle type (!mlar.compute, "
                           "!mlar.memory, etc.)");
  }

  // Parse comma separator
  if (parser.parseComma())
    return failure();

  // Parse target operand with optional type
  OpAsmParser::UnresolvedOperand target;
  Type targetType;
  bool hasTargetType = false;
  if (parseOperandWithType(target, targetType, hasTargetType).failed())
    return failure();

  // Validate target type if provided
  if (hasTargetType && !isValidHandleType(targetType)) {
    return parser.emitError(parser.getNameLoc(),
                           "target type must be a mlar handle type (!mlar.compute, "
                           "!mlar.memory, etc.)");
  }

  // Parse optional indices (comma-separated list after target)
  SmallVector<OpAsmParser::UnresolvedOperand> indices;
  if (parser.parseOptionalComma().succeeded()) {
    // Try to parse indices - if next token is '{', parseOperandList will fail
    // gracefully and we'll continue to parse attributes
    (void)parser.parseOperandList(indices);
  }

  // Parse attributes dictionary (map, bandwidth, etc.)
  if (parser.parseOptionalAttrDict(result.attributes).failed())
    return failure();

  // Parse optional result type annotation
  Type resultType;
  if (succeeded(parser.parseOptionalColon())) {
    if (parser.parseType(resultType).failed())
      return failure();
  } else {
    // No result type provided, use default
    resultType = InterconnectHandleType::get(context);
  }

  // Resolve operand types - infer from operands if not explicitly provided
  // Build type list and resolve operands
  // For typed operands, add a type; for untyped ones, this will cause MLI to infer
  // First build all types
  SmallVector<Type> allTypes;
  if (hasSourceType) {
    allTypes.push_back(sourceType);
  }
  if (hasTargetType) {
    allTypes.push_back(targetType);
  }
  
  // Resolve source and target together if both have types, or individually if only one does
 if (hasSourceType && hasTargetType) {
    // Both have types
    SmallVector<Type> types = {sourceType, targetType};
    SmallVector<OpAsmParser::UnresolvedOperand> operands = {source, target};
    if (parser.resolveOperands(operands, types, parser.getNameLoc(), result.operands))
      return failure();
  } else if (hasSourceType) {
    // Only source has type
    if (parser.resolveOperands({source}, {sourceType}, parser.getNameLoc(), result.operands))
      return failure();
    if (parser.resolveOperands({target}, {}, parser.getNameLoc(), result.operands))
      return failure();
  } else if (hasTargetType) {
    // Only target has type
    if (parser.resolveOperands({source}, {}, parser.getNameLoc(), result.operands))
      return failure();
    if (parser.resolveOperands({target}, {targetType}, parser.getNameLoc(), result.operands))
      return failure();
  } else {
    // Neither has type - infer from SSA defs
    if (parser.resolveOperands({source, target}, {}, parser.getNameLoc(), result.operands))
      return failure();
  }

  // Resolve indices (all should be index type)
  if (!indices.empty()) {
    SmallVector<Type> indexTypes(indices.size(), builder.getIndexType());
    if (parser.resolveOperands(indices, indexTypes, parser.getNameLoc(), result.operands))
      return failure();
  }

  // Set result type
  result.addTypes(resultType);

  return success();
}

/// Print InterconnectsOp with custom syntax
void InterconnectsOp::print(OpAsmPrinter &p) {
  p << " ";
  
  // Print symbol name
  p << "\"";
  p << getSymName();
  p << "\" ";
  
  p.printOperand(getSource());
  p << " : ";
  p.printType(getSource().getType());
  p << ", ";
  p.printOperand(getTarget());
  p << " : ";
  p.printType(getTarget().getType());
  
  if (!getIndices().empty()) {
    p << ", ";
    p.printOperands(getIndices());
  }
  
  p << " ";
  // Elide sym_name from attribute dict since it's printed separately
  // spatial_dims will be printed in the attribute dict automatically
  SmallVector<StringRef> elidedAttrs = {"sym_name"};
  p.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  
  p << " : ";
  p.printType(getHandle().getType());
}
