"""LaTeX → MathML → Word OMML 转换。

目标：把毕业论文里 $...$ / $$...$$ 公式转成 Word 原生数学对象（OMML），
让 Word 双击可编辑，显示真正的数学符号（σ、∑、分数线、上下标）。

输入子集（论文实际用到的）：
- mrow, mi, mn, mo, mtext
- msub, msup, msubsup
- mfrac
- msqrt（兜底）
- mathvariant="normal"（\mathrm 函数名转出来的，需要直立体）
- 其余未知元素回退为纯文本

不支持的、需要在 LaTeX 层预处理：
- \tag{X-Y}：在调用前抽出来，作为段落右侧的编号字符串单独返回
"""
from __future__ import annotations

import re
from typing import Optional, Tuple

from lxml import etree

from latex2mathml.converter import convert as latex_to_mathml


MATH_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
MML_NS = "http://www.w3.org/1998/Math/MathML"
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML_NS = "http://www.w3.org/XML/1998/namespace"


def _m(tag: str) -> etree._Element:
    return etree.SubElement(etree.Element(f"{{{MATH_NS}}}__root"), f"{{{MATH_NS}}}{tag}")


def _new(tag: str) -> etree._Element:
    return etree.Element(f"{{{MATH_NS}}}{tag}", nsmap={"m": MATH_NS})


def _qname(ns: str, name: str) -> str:
    return f"{{{ns}}}{name}"


def _mk_run(text: str, plain: bool = False) -> etree._Element:
    """构造一个 m:r（含 m:t 文本）。plain=True 时设置 sty=p（直立体，用于函数名）"""
    r = _new("r")
    if plain:
        rpr = etree.SubElement(r, _qname(MATH_NS, "rPr"))
        sty = etree.SubElement(rpr, _qname(MATH_NS, "sty"))
        sty.set(_qname(MATH_NS, "val"), "p")
    t = etree.SubElement(r, _qname(MATH_NS, "t"))
    t.set(_qname(XML_NS, "space"), "preserve")
    t.text = text
    return t.getparent()


def _local(elem: etree._Element) -> str:
    return etree.QName(elem).localname


def _convert(node: etree._Element) -> list:
    """把一个 MathML 节点转成一组 OMML 元素（列表，方便 mrow 直接展开）"""
    tag = _local(node)

    if tag == "math":
        out = []
        for c in node:
            out.extend(_convert(c))
        return out

    if tag == "mrow":
        out = []
        for c in node:
            out.extend(_convert(c))
        return out

    if tag in ("mi", "mn", "mo", "mtext"):
        text = node.text or ""
        if not text:
            return []
        plain = (tag == "mi" and node.get("mathvariant") == "normal")
        return [_mk_run(text, plain=plain)]

    if tag == "msub":
        children = list(node)
        if len(children) != 2:
            return _fallback_text(node)
        ssub = _new("sSub")
        e = etree.SubElement(ssub, _qname(MATH_NS, "e"))
        for x in _convert(children[0]):
            e.append(x)
        sub = etree.SubElement(ssub, _qname(MATH_NS, "sub"))
        for x in _convert(children[1]):
            sub.append(x)
        return [ssub]

    if tag == "msup":
        children = list(node)
        if len(children) != 2:
            return _fallback_text(node)
        ssup = _new("sSup")
        e = etree.SubElement(ssup, _qname(MATH_NS, "e"))
        for x in _convert(children[0]):
            e.append(x)
        sup = etree.SubElement(ssup, _qname(MATH_NS, "sup"))
        for x in _convert(children[1]):
            sup.append(x)
        return [ssup]

    if tag == "msubsup":
        children = list(node)
        if len(children) != 3:
            return _fallback_text(node)
        node3 = _new("sSubSup")
        e = etree.SubElement(node3, _qname(MATH_NS, "e"))
        for x in _convert(children[0]):
            e.append(x)
        sub = etree.SubElement(node3, _qname(MATH_NS, "sub"))
        for x in _convert(children[1]):
            sub.append(x)
        sup = etree.SubElement(node3, _qname(MATH_NS, "sup"))
        for x in _convert(children[2]):
            sup.append(x)
        return [node3]

    if tag == "mfrac":
        children = list(node)
        if len(children) != 2:
            return _fallback_text(node)
        f = _new("f")
        num = etree.SubElement(f, _qname(MATH_NS, "num"))
        for x in _convert(children[0]):
            num.append(x)
        den = etree.SubElement(f, _qname(MATH_NS, "den"))
        for x in _convert(children[1]):
            den.append(x)
        return [f]

    if tag == "msqrt":
        rad = _new("rad")
        rpr = etree.SubElement(rad, _qname(MATH_NS, "radPr"))
        dh = etree.SubElement(rpr, _qname(MATH_NS, "degHide"))
        dh.set(_qname(MATH_NS, "val"), "1")
        etree.SubElement(rad, _qname(MATH_NS, "deg"))
        e = etree.SubElement(rad, _qname(MATH_NS, "e"))
        for c in node:
            for x in _convert(c):
                e.append(x)
        return [rad]

    if tag == "mroot":
        children = list(node)
        if len(children) != 2:
            return _fallback_text(node)
        rad = _new("rad")
        deg = etree.SubElement(rad, _qname(MATH_NS, "deg"))
        for x in _convert(children[1]):
            deg.append(x)
        e = etree.SubElement(rad, _qname(MATH_NS, "e"))
        for x in _convert(children[0]):
            e.append(x)
        return [rad]

    return _fallback_text(node)


def _fallback_text(node: etree._Element) -> list:
    """未知元素：递归取所有 text，作为一个普通 run。"""
    text = "".join(node.itertext())
    if text.strip():
        return [_mk_run(text)]
    return []


_TAG_RE = re.compile(r"\\tag\{([^}]+)\}")


def split_tag(latex: str) -> Tuple[str, Optional[str]]:
    """从 LaTeX 字符串中抽出 \\tag{X-Y}，返回 (无tag的latex, 编号文本 or None)"""
    m = _TAG_RE.search(latex)
    if not m:
        return latex, None
    tag = f"({m.group(1)})"
    cleaned = _TAG_RE.sub("", latex).strip()
    return cleaned, tag


def latex_to_omath(latex: str, display: bool = False) -> etree._Element:
    """把 LaTeX 字符串转为 m:oMath 元素（已去掉 \\tag）。display 仅用于 latex2mathml 的提示参数。"""
    mathml_str = latex_to_mathml(latex, display="block" if display else "inline")
    # 解析 MathML
    parser = etree.XMLParser(remove_blank_text=False)
    mml = etree.fromstring(mathml_str.encode("utf-8"), parser)
    # 构造 m:oMath
    omath = _new("oMath")
    for elem in _convert(mml):
        omath.append(elem)
    return omath


def latex_to_omath_para(latex: str) -> Tuple[etree._Element, Optional[str]]:
    """块公式：返回 (m:oMathPara 元素, 编号文本)。编号要由调用方插入到段落右侧。"""
    cleaned, tag = split_tag(latex)
    omath = latex_to_omath(cleaned, display=True)
    omath_para = _new("oMathPara")
    omath_para.append(omath)
    return omath_para, tag


if __name__ == "__main__":
    import sys
    cases = [
        r"F \in \mathbb{R}^{C \times H \times W}",
        r"M_c(F) = \sigma\bigl(\mathrm{MLP}(\mathrm{AvgPool}(F))\bigr) \tag{2-1}",
        r"\mathrm{EAR} = \frac{\|p_1 - p_5\| + \|p_2 - p_4\|}{2 \cdot \|p_0 - p_3\|}",
        r"\mathrm{PERCLOS} = \frac{\sum_{i=t-N+1}^{t} \mathbb{1}[\mathrm{EAR}_i < \theta]}{N}",
    ]
    for latex in cases:
        print("LATEX:", latex)
        cleaned, tag = split_tag(latex)
        omath = latex_to_omath(cleaned)
        xml = etree.tostring(omath, pretty_print=True, encoding="unicode")
        print(xml)
        if tag:
            print("TAG:", tag)
        print("---")
