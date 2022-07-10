from __future__ import annotations

import math

from jax import tree_flatten

from kernex.treeclass.utils import (
    is_treeclass,
    is_treeclass_leaf,
    leaves_param_count,
    leaves_param_count_and_size,
    leaves_param_format,
    node_class_name,
    node_format,
    sequential_model_shape_eval,
)


def resolve_line(cols):
    """
    === Explanation
        combine columns of single line by merging their borders
    
    === Examples
        >>> resolve_line(['ab','b│','│c'])
        'abb│c'

        >>> resolve_line(['ab','b┐','┌c'])
        'abb┬c'    
    
    """

    cols = list(map(list, cols))  # convert each col to col of chars
    alpha = ["│", "┌", "┐", "└", "┘", "┤", "├"]

    for index in range(len(cols) - 1):

        if cols[index][-1] == "┐" and cols[index + 1][0] in ["┌", "─"]:
            cols[index][-1] = "┬"
            cols[index + 1].pop(0)

        elif cols[index][-1] == "┘" and cols[index + 1][0] in ["└", "─"]:
            cols[index][-1] = "┴"
            cols[index + 1].pop(0)

        elif cols[index][-1] == "┤" and cols[index + 1][0] in ["├", "─", "└"
                                                               ]:  #
            cols[index][-1] = "┼"
            cols[index + 1].pop(0)

        elif cols[index][-1] in ["┘", "┐", "─"
                                 ] and cols[index + 1][0] in ["├"]:
            cols[index][-1] = "┼"
            cols[index + 1].pop(0)

        elif cols[index][-1] == "─" and cols[index + 1][0] == "└":
            cols[index][-1] = "┴"
            cols[index + 1].pop(0)

        elif cols[index][-1] == "─" and cols[index + 1][0] == "┌":
            cols[index][-1] = "┬"
            cols[index + 1].pop(0)

        elif cols[index][-1] == "│" and cols[index + 1][0] == "─":
            cols[index][-1] = "├"
            cols[index + 1].pop(0)

        elif cols[index][-1] == " ":
            cols[index].pop()

        elif cols[index][-1] in alpha and cols[index + 1][0] in [*alpha, " "]:
            cols[index + 1].pop(0)

    return ''.join(map(lambda x: ''.join(x), cols))


def hbox(*text):

    boxes = list(map(vbox, text))
    boxes = [(box).split('\n') for box in boxes]
    max_col_height = max([len(b) for b in boxes])
    boxes = [b + [' ' * len(b[0])] * (max_col_height - len(b)) for b in boxes]
    fmt = ''

    for i, line in enumerate(zip(*boxes)):
        fmt += resolve_line(line) + '\n'
    return fmt


def hstack(boxes):

    boxes = [(box).split('\n') for box in boxes]
    max_col_height = max([len(b) for b in boxes])

    # expand height of each col before merging
    boxes = [b + [' ' * len(b[0])] * (max_col_height - len(b)) for b in boxes]

    fmt = ''

    cells = tuple(zip(*boxes))

    for i, line in enumerate(cells):
        fmt += resolve_line(line) + ('\n' if i != (len(cells) - 1) else '')

    return fmt


def vbox(*text):
    """
    === Explanation
        create vertically stacked text boxes
    
    === Examples

        >> vbox("a","b")
        ┌───┐
        │a  │
        ├───┤
        │b  │
        └───┘
        
        >> vbox("a","","a")
        ┌───┐
        │a  │
        ├───┤
        │   │
        ├───┤
        │a  │
        └───┘
        
        >> vbox("a","","a",header=True)
        ┏━━━┓
        ┃a  ┃
        ┡━━━┩
        │   │
        ├───┤
        │a  │
        └───┘
    """

    max_width = max(
        tree_flatten([[len(t) for t in item.split('\n')]
                      for item in text])[0]) + 0

    top = f"┌{'─'*max_width}┐"
    line = f"├{'─'*max_width}┤"
    side = [
        '\n'.join([f"│{t}{' '*(max_width-len(t))}│" for t in item.split('\n')])
        for item in text
    ]
    btm = f"└{'─'*max_width}┘"

    formatted = ''

    for i, s in enumerate(side):

        if i == 0:
            formatted += f"{top}\n{s}\n{line if len(side)>1 else btm}"

        elif i == len(side) - 1:
            formatted += f"\n{s}\n{btm}"

        else:
            formatted += f"\n{s}\n{line}"

    return formatted


def table(lines):
    """
    
    === Explanation 
        create a table with self aligning rows and cols
    
    === Args
        lines : list of lists of cols values

    === Examples
        >>> print(table([['1\n','2'],['3','4000']]))
            ┌─┬────────┐
            │1│3       │
            │ │        │
            ├─┼────────┤
            │2│40000000│
            └─┴────────┘
    
    
    """
    # align cells vertically
    for i, cells in enumerate(zip(*lines)):
        max_cell_height = max(map(lambda x: x.count('\n'), cells))
        for j in range(len(cells)):
            lines[j][i] += '\n' * (max_cell_height - lines[j][i].count('\n'))
    cols = [vbox(*col) for col in lines]

    return hstack(cols)


def layer_box(name, indim=None, outdim=None):
    """
    === Explanation
        create a keras-like layer diagram
    
    ==== Examples
        >>> print(layer_box("Test",(1,1,1),(1,1,1)))
        ┌──────┬────────┬───────────┐
        │      │ Input  │ (1, 1, 1) │
        │ Test │────────┼───────────┤
        │      │ Output │ (1, 1, 1) │
        └──────┴────────┴───────────┘

    """

    return hstack([
        vbox(f"\n {name} \n"),
        table([[" Input ", " Output "], [f" {indim} ", f" {outdim} "]])
    ])


def tree_box(model, array=None):
    """
    === plot tree classes 
    """

    def recurse(model, parent_name):

        nonlocal shapes

        if is_treeclass_leaf(model):
            box = layer_box(
                f"{model.__class__.__name__}({parent_name})",
                node_format(shapes[0]) if array is not None else None,
                node_format(shapes[1]) if array is not None else None)

            if shapes is not None:
                shapes.pop(0)
            return box

        else:
            level_nodes = []

            for field in model.__dataclass_fields__.values():
                cur_node = model.__dict__[field.name]

                if is_treeclass(cur_node):
                    level_nodes += [f"{recurse(cur_node,field.name)}"]

                else:
                    level_nodes += [
                        vbox(f"{field.name}={node_format(cur_node)}")
                    ]

            return vbox(f"{model.__class__.__name__}({parent_name})",
                        '\n'.join(level_nodes))

    shapes = sequential_model_shape_eval(model,
                                         array) if array is not None else None
    return recurse(model, "Parent")


def tree_diagram(model):
    """
    === Explanation 
        pretty print treeclass model with tree structure diagram
    
    === Args
        tree : boolean to create tree-structure 
    """

    def recurse(model, parent_level_count):

        nonlocal fmt

        if is_treeclass(model):

            cur_children_count = len(model.__dataclass_fields__)

            for i, field in enumerate(model.__dataclass_fields__.values()):
                cur_node = model.__dict__[field.name]

                fmt += '\n' + ''.join([(("│" if lvl > 1 else "") + "\t")
                                       for lvl in parent_level_count])

                if is_treeclass(cur_node):

                    layer_class_name = cur_node.__class__.__name__

                    fmt += ("├── " if i < (cur_children_count - 1) else
                            "└──") + f"{field.name}={layer_class_name}"
                    recurse(cur_node,
                            parent_level_count + [cur_children_count - i])

                else:
                    fmt += ("├── " if i < (cur_children_count - 1) else "└── ")
                    fmt += f"{field.name}={node_format(cur_node)}"
                    recurse(cur_node, parent_level_count + [1])

            fmt += '\t'

    fmt = f"{(model.__class__.__name__)}"

    recurse(model, [1])

    return fmt.expandtabs(4)


def tree_indent(model):
    """
    === Explanation 
        pretty print treeclass model with indentation
    
    === Args
        tree : boolean to create tree-structure 
    """

    def recurse(model, parent_level_count):

        nonlocal fmt

        if is_treeclass(model):
            cur_children_count = len(model.__dataclass_fields__)

            for i, field in enumerate(model.__dataclass_fields__.values()):
                cur_node = model.__dict__[field.name]

                fmt += '\n' + "\t" * len(parent_level_count)

                if is_treeclass(cur_node):

                    layer_class_name = f"{cur_node.__class__.__name__}"
                    fmt += f"{field.name}={layer_class_name}" + "("

                    recurse(cur_node,
                            parent_level_count + [cur_children_count - i])

                else:
                    fmt += f"{field.name}={node_format(cur_node)}" + (
                        "" if i < (cur_children_count - 1) else ")")
                    recurse(cur_node, parent_level_count + [1])

    fmt = f"{(model.__class__.__name__)}("
    recurse(model, [1])
    fmt += ")"

    return fmt.expandtabs(4)


def summary(model, array=None) -> str:
    """
    === Explanation
        return a printable string containing summary of treeclass leaves.
    
    === Example:


    
    """

    dynamic_leaves = [leave.tree_fields[0] for leave in model.treeclass_leaves]

    leaves_name = [node_class_name(leaf) for leaf in model.treeclass_leaves]
    params_count, params_size = zip(
        *leaves_param_count_and_size(dynamic_leaves))
    params_repr = leaves_param_format(dynamic_leaves)

    total_param_count = 0
    total_param_size = 0

    if array is not None:
        params_shape = sequential_model_shape_eval(model, array)
    else:
        params_shape = [None] * len(dynamic_leaves)

    ROW = [["Type ", "Output ", "Param #", "Size ", "Config"]]

    order_kw = ['B', 'KB', 'MB', 'GB']

    for (pname, pcount, psize, prepr,
         pshape) in (zip(leaves_name, params_count, params_size, params_repr,
                         params_shape)):

        cur_type = f"{pname}"
        cur_count = f"{int(pcount.real+pcount.imag):,}"
        cur_size = (psize.real + psize.imag)
        cur_size_order = int(math.log(cur_size, 1024)) if cur_size > 0 else 0
        cur_size = f"{(cur_size)/(1024**cur_size_order):.3f} {order_kw[cur_size_order]}"
        cur_repr = "\n".join([f" {k}={v}"
                              for k, v in prepr.items()]).replace(" ", "")
        cur_shape = node_format(pshape) if array is not None else None

        total_param_count += pcount
        total_param_size += psize

        ROW += [[cur_type, cur_shape, cur_count, cur_size, cur_repr]]

    COL = [list(c) for c in zip(*ROW)]

    if array is None:
        COL.pop(1)

    layer_table = table(COL)

    table_width = len(layer_table.split('\n')[0])

    # summary row
    total_dparam = total_param_count.real
    total_sparam = total_param_count.imag

    total_dparam_size = total_param_size.real
    dorder = int(math.log(total_dparam_size,
                          1024)) if total_dparam_size > 0 else 0  #

    total_sparam_size = total_param_size.imag
    sorder = int(math.log(total_sparam_size,
                          1024)) if total_sparam_size > 0 else 0  #

    total_param_size = total_dparam_size + total_sparam_size
    torder = int(math.log(total_param_size,
                          1024)) if total_param_size > 0 else 0  #

    param_summary = (
        f"Total params :\t{int(total_dparam+total_sparam):,}\n"
        f"Inexact params:\t{int(total_dparam):,}\n"
        f"Other params:\t{int(total_sparam):,}\n"
        f"{'-'*table_width}\n"
        f"Total size :\t{total_param_size/(1024**torder):.3f} {order_kw[torder]}\n"
        f"Inexact size:\t{total_dparam_size/(1024**dorder):.3f} {order_kw[dorder]}\n"
        f"Other size:\t{total_sparam_size/(1024**sorder):.3f} {order_kw[sorder]}\n"
        f"{'='*table_width}")

    return layer_table + '\n' + param_summary
