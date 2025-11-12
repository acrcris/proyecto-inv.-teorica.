import ast
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional


def dotted_from_attribute(node: ast.AST) -> Optional[str]:
    # Build dotted name from nested Attribute/Name nodes
    parts: List[str] = []
    curr = node
    while isinstance(curr, ast.Attribute):
        parts.append(curr.attr)
        curr = curr.value
    if isinstance(curr, ast.Name):
        parts.append(curr.id)
        return ".".join(reversed(parts))
    return None


class ClassInfo:
    def __init__(self, module: str, name: str):
        self.module = module  # e.g., source.layers.klayer
        self.name = name      # e.g., KLayer
        self.bases: Set[str] = set()   # fully qualified internal class names
        self.uses: Set[str] = set()    # fully qualified internal class names

    @property
    def fqname(self) -> str:
        return f"{self.module}.{self.name}"


def collect_python_files(src_root: Path) -> List[Path]:
    return [p for p in src_root.rglob("*.py")]


def module_name_for(path: Path, src_root: Path) -> str:
    # Convert file path to module name rooted at 'source'
    rel = path.relative_to(src_root)
    parts = rel.with_suffix("").parts
    return ".".join((src_root.name,) + parts)


def parse_classes(src_root: Path) -> Dict[str, ClassInfo]:
    classes: Dict[str, ClassInfo] = {}
    for py in collect_python_files(src_root):
        mod = module_name_for(py, src_root)
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                ci = ClassInfo(mod, node.name)
                classes[ci.fqname] = ci
    return classes


def build_import_aliases(tree: ast.AST) -> Dict[str, str]:
    # Map local alias -> fully qualified module or symbol
    aliases: Dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("source"):
                for alias in node.names:
                    local = alias.asname or alias.name
                    aliases[local] = f"{node.module}.{alias.name}"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("source"):
                    local = alias.asname or alias.name.split(".")[-1]
                    aliases[local] = alias.name
    return aliases


def resolve_internal_target(name: str, aliases: Dict[str, str], all_classes: Dict[str, ClassInfo], local_module: Optional[str] = None) -> Optional[str]:
    # Try direct alias (module+symbol), dotted, or by class name fallback
    # 1) If name refers to a local class in the same module
    if local_module is not None and name.isidentifier():
        candidate = f"{local_module}.{name}"
        if candidate in all_classes:
            return candidate

    # 2) If name contains dots and matches a known class fqname
    if "." in name and name in all_classes:
        return name
    # 3) If alias maps name to module path; expand to module+Class
    if name in aliases:
        aliased = aliases[name]
        # If alias already points to a class fqname
        if aliased in all_classes:
            return aliased
        # Otherwise alias points to module; try module + name
        candidate = f"{aliased}.{name}"
        if candidate in all_classes:
            return candidate
    # 4) Fallback: match by class short name (may be ambiguous)
    matches = [fq for fq in all_classes.keys() if fq.endswith("." + name)]
    if len(matches) == 1:
        return matches[0]
    return None


def extract_relationships(src_root: Path, classes: Dict[str, ClassInfo]) -> None:
    # Walk each file again to map bases and uses
    for py in collect_python_files(src_root):
        mod = module_name_for(py, src_root)
        tree = ast.parse(py.read_text(encoding="utf-8"))
        aliases = build_import_aliases(tree)

        # Build mapping for classes in this module
        local_classes = {c.name: c for c in classes.values() if c.module == mod}

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                ci = local_classes.get(node.name)
                if not ci:
                    continue
                # Inheritance relationships
                for base in node.bases:
                    base_name = None
                    if isinstance(base, ast.Name):
                        base_name = base.id
                    elif isinstance(base, ast.Attribute):
                        base_name = dotted_from_attribute(base)
                    if base_name:
                        target = resolve_internal_target(base_name, aliases, classes, local_module=mod)
                        if target:
                            ci.bases.add(target)

                # Usage/composition relationships: scan within class body
                for inner in ast.walk(node):
                    if isinstance(inner, ast.Call):
                        fn_name: Optional[str] = None
                        if isinstance(inner.func, ast.Name):
                            fn_name = inner.func.id
                        elif isinstance(inner.func, ast.Attribute):
                            fn_name = dotted_from_attribute(inner.func)
                        if fn_name:
                            target = resolve_internal_target(fn_name, aliases, classes, local_module=mod)
                            if target:
                                if target != ci.fqname:
                                    ci.uses.add(target)


def emit_dot(classes: Dict[str, ClassInfo]) -> str:
    lines: List[str] = []
    lines.append("digraph G {")
    lines.append("  rankdir=LR;")
    lines.append("  node [shape=box, style=rounded];")

    # Define nodes
    for fq, ci in sorted(classes.items()):
        label = f"{ci.name}\n[{ci.module}]"
        lines.append(f'  "{fq}" [label="{label}"];')

    # Edges: inheritance
    for fq, ci in sorted(classes.items()):
        for b in sorted(ci.bases):
            lines.append(f'  "{fq}" -> "{b}" [label="hereda"];')

    # Edges: usage/composition
    for fq, ci in sorted(classes.items()):
        for u in sorted(ci.uses):
            lines.append(f'  "{fq}" -> "{u}" [label="usa"];')

    lines.append("}")
    return "\n".join(lines)


def main():
    # Repository structure: this file is under akorn/scripts/
    repo_root = Path(__file__).resolve().parents[2]
    src_root = repo_root / "akorn" / "source"
    if not src_root.exists():
        raise SystemExit(f"source root not found: {src_root}")

    classes = parse_classes(src_root)
    extract_relationships(src_root, classes)
    dot = emit_dot(classes)

    out_path = repo_root / "alternativa1" / "Relaciones_Source.dot"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(dot, encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
