"""把 毕业论文.docx 里所有 LaTeX 残留转成 Word 原生 OMML 公式。

- 块公式（整段是裸 LaTeX，末尾可能带 "(X-Y)" 编号）：整段清空 → 居中 → 插 oMathPara → 右侧编号
- 行内 $...$：段落 runs 重建，文本段用普通 run，LaTeX 段用 OMath
- 保留段落 pPr 属性与中文宋体+Times New Roman 字体
"""
import re
import sys
sys.path.insert(0, 'scripts')

from docx import Document
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.enum.text import WD_ALIGN_PARAGRAPH
from lxml import etree

from mathml_to_omml import latex_to_omath, latex_to_omath_para

INLINE_PAT = re.compile(r'\$([^$]+)\$')
TAIL_TAG_PAT = re.compile(r'\s*\(\s*([0-9]+\s*[-–]\s*[0-9]+)\s*\)\s*$')


def is_block_formula(text):
    """段落是裸 LaTeX 块公式：无中文、无 $$、含反斜杠命令或孤立数学符号"""
    if '$' in text:
        return False
    if any('一' <= c <= '鿿' for c in text):
        return False
    if re.search(r'\\(frac|sigma|theta|sum|mathrm|mathbb|mathcal|alpha|beta|max|min|cdot|times|leq|geq|in)\b', text):
        return True
    return False


def make_text_run(text):
    """创建带中文宋体+TNR 字体的 w:r"""
    r = OxmlElement('w:r')
    rpr = OxmlElement('w:rPr')
    rf = OxmlElement('w:rFonts')
    rf.set(qn('w:ascii'), 'Times New Roman')
    rf.set(qn('w:hAnsi'), 'Times New Roman')
    rf.set(qn('w:eastAsia'), '宋体')
    rf.set(qn('w:cs'), 'Times New Roman')
    rpr.append(rf)
    sz = OxmlElement('w:sz')
    sz.set(qn('w:val'), '24')
    rpr.append(sz)
    r.append(rpr)
    t = OxmlElement('w:t')
    t.set(qn('xml:space'), 'preserve')
    t.text = text
    r.append(t)
    return r


def get_or_add_ppr(p_el):
    ppr = p_el.find(qn('w:pPr'))
    if ppr is None:
        ppr = OxmlElement('w:pPr')
        p_el.insert(0, ppr)
    return ppr


def set_center(p_el):
    ppr = get_or_add_ppr(p_el)
    jc = ppr.find(qn('w:jc'))
    if jc is None:
        jc = OxmlElement('w:jc')
        ppr.append(jc)
    jc.set(qn('w:val'), 'center')


def clear_runs(p_el):
    """清空段落里除 pPr 之外的所有 child"""
    for child in list(p_el):
        if child.tag != qn('w:pPr'):
            p_el.remove(child)


def main():
    doc = Document('毕业论文.docx')

    block_ok = block_fail = 0
    inline_ok = inline_fail = 0

    for para in doc.paragraphs:
        text = para.text
        if not text:
            continue

        # --- 块公式 ---
        if is_block_formula(text):
            m = TAIL_TAG_PAT.search(text)
            if m:
                tag_text = f"({m.group(1).replace(' ', '')})"
                latex = TAIL_TAG_PAT.sub('', text).strip()
            else:
                tag_text = None
                latex = text.strip()

            try:
                omath_para, _ = latex_to_omath_para(latex)
            except Exception as e:
                print(f"  ❌ 块: {latex[:50]}... -> {type(e).__name__}: {str(e)[:60]}")
                block_fail += 1
                continue

            p_el = para._element
            clear_runs(p_el)
            set_center(p_el)
            p_el.append(omath_para)
            if tag_text:
                # 中间空白 + 编号
                spacer = make_text_run('    ')  # 4 个 NBSP
                p_el.append(spacer)
                p_el.append(make_text_run(tag_text))
            block_ok += 1
            continue

        # --- 行内 $...$ ---
        matches = list(INLINE_PAT.finditer(text))
        if not matches:
            continue

        segments = []
        last_end = 0
        for m in matches:
            if m.start() > last_end:
                segments.append(('text', text[last_end:m.start()]))
            segments.append(('math', m.group(1)))
            last_end = m.end()
        if last_end < len(text):
            segments.append(('text', text[last_end:]))

        # 预先把 LaTeX 都转一遍，全部成功才动手
        omaths = []
        ok = True
        for kind, content in segments:
            if kind == 'math':
                try:
                    omaths.append(latex_to_omath(content))
                except Exception as e:
                    print(f"  ❌ 行内: ${content}$ -> {type(e).__name__}: {str(e)[:60]}")
                    ok = False
                    break
        if not ok:
            inline_fail += 1
            continue

        # 重建
        p_el = para._element
        clear_runs(p_el)
        oi = 0
        for kind, content in segments:
            if kind == 'text':
                p_el.append(make_text_run(content))
            else:
                p_el.append(omaths[oi])
                oi += 1
        inline_ok += 1

    doc.save('毕业论文.docx')
    print(f"\n块公式：成功 {block_ok}，失败 {block_fail}")
    print(f"行内公式段：成功 {inline_ok}，失败 {inline_fail}")


if __name__ == '__main__':
    main()
