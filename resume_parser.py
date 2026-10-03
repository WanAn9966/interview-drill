"""Local, source-preserving resume parsing: document -> sections -> entries.

Headings carry context across wrapped lines. Ambiguous text stays in '其他'
instead of becoming an invented project. No model or API key is required.
"""
import io
import re
import unicodedata
import zipfile
from pathlib import Path


KINDS = ('项目', '实习经历', '学历', '工作技能', '工作经历', '其他')
ALIASES = {
    '项目': ('项目经历', '项目经验', '个人项目', '科研项目', '项目实践', '项目',
             'project experience', 'personal projects', 'selected projects', 'projects'),
    '实习经历': ('实习经历', '实习经验', '实习实践', '实习', 'internship experience', 'internships', 'internship'),
    '学历': ('教育背景', '教育经历', '学习经历', '学历信息', '学历', '教育', 'education', 'academic background'),
    '工作技能': ('专业技能', '技术技能', '职业技能', '工作技能', '技能清单', '技能特长', '个人技能', '技能',
                 'technical skills', 'professional skills', 'core competencies', 'skills'),
    '工作经历': ('工作经历', '工作经验', '职业经历', '任职经历', 'employment history', 'work experience', 'professional experience'),
    '其他': ('个人信息', '基本信息', '个人简介', '自我评价', '荣誉奖项', '获奖经历', '证书', '校园经历',
             '求职意向', '联系方式', 'summary', 'profile', 'awards', 'certifications', 'contact', 'experience'),
}


def alias_pattern(alias):
    if re.search('[\u4e00-\u9fff]', alias):
        return r'\s*'.join(map(re.escape, alias))
    return r'\s+'.join(map(re.escape, alias.split()))


HEADINGS = [(kind, re.compile(r'^' + alias_pattern(alias) + r'(?=$|\s|[:：|丨\]】])', re.I))
            for kind, aliases in ALIASES.items() for alias in sorted(aliases, key=len, reverse=True)]
INLINE = re.compile(r'(?<=[;；。])\s*(?=(?:' + '|'.join(
    alias_pattern(a) for aliases in ALIASES.values() for a in aliases) + r')\s*[:：])', re.I)
DATE_RANGE = re.compile(r'(?:19|20)\d{2}[./年-]\d{1,2}(?:月)?\s*(?:-|—|–|~|～|至|to)\s*(?:(?:19|20)\d{2}|至今|现在|present)', re.I)


class ResumeError(ValueError):
    pass


def normalize(text):
    text = unicodedata.normalize('NFKC', text).replace('\r\n', '\n').replace('\r', '\n')
    return text.replace('\x00', '').replace('\u200b', '').replace('\ufeff', '').replace('\xa0', ' ')


def heading(line):
    candidate = re.sub(r'^\s*(?:#{1,6}\s*|[●•◆▪■★]\s*|[一二三四五六七八九十\d]+[、.)．]\s*)', '', line)
    candidate = candidate.strip().strip('*').lstrip('【[').strip()
    for kind, pattern in HEADINGS:
        match = pattern.match(candidate)
        if match:
            remainder = candidate[match.end():].lstrip(' \t:：|丨】]—-').strip('* ')
            # Avoid treating prose beginning '项目 ...' as a section title.
            if remainder and not re.match(r'^\s*[:：|丨\]】]', candidate[match.end():]) and len(candidate) > 90:
                continue
            return kind, remainder
    return None


def _columns(layout):
    """Split a stable gutter only when both sides contain section headings.

    A right-aligned date column must not be mistaken for a second text column.
    Unusual mixed layouts are deliberately left for manual review.
    """
    lines = layout.splitlines()
    gaps = []
    for line in lines:
        for match in re.finditer(r'(?<=\S) {5,}(?=\S)', line):
            gaps.append((match.start(), match.end()))
    if not gaps:
        return layout, False
    candidates = sorted({(a+b)//2 for a,b in gaps})
    for split in sorted(candidates, key=lambda x: -sum(a <= x <= b for a,b in gaps)):
        left, right = [], []
        crossing = 0
        for line in lines:
            if len(line) > split and line[:split].strip() and line[split:].strip():
                if line[split-1:split+1] != '  ':
                    crossing += 1
            left.append(line[:split].rstrip())
            right.append(line[split:].strip())
        if crossing == 0 and any(heading(x.strip()) for x in left) and any(heading(x) for x in right):
            return '\n'.join(left) + '\n\n' + '\n'.join(right), True
    return layout, False


def extract_document(name, blob):
    if len(blob) > 8 * 1024 * 1024:
        raise ResumeError('文件最多8MB，请压缩后再上传')
    ext = Path(name).suffix.lower()
    warnings = []
    if ext == '.pdf':
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(blob))
        if reader.is_encrypted and not reader.decrypt(''):
            raise ResumeError('PDF有密码保护，请解密后上传')
        if len(reader.pages) > 30:
            raise ResumeError('请上传30页以内的简历')
        pages = []
        missing = []
        for i, page in enumerate(reader.pages, 1):
            layout = page.extract_text(extraction_mode='layout') or ''
            text, split = _columns(layout)
            if split:
                warnings.append(f'第{i}页按左右两栏提取，请核对章节顺序。')
            if len(re.sub(r'\s', '', text)) < 10:
                missing.append(i)
            pages.append(text)
        if missing:
            warnings.append('第' + '、'.join(map(str, missing)) + '页文字不足，可能是扫描图片；请先OCR或补充粘贴文本。')
        return '\f'.join(pages), warnings
    if ext == '.docx':
        with zipfile.ZipFile(io.BytesIO(blob)) as archive:
            if sum(i.file_size for i in archive.infolist()) > 32 * 1024 * 1024:
                raise ResumeError('文档解压后过大，请精简后上传')
        from docx import Document
        from docx.table import Table
        from docx.text.paragraph import Paragraph
        doc = Document(io.BytesIO(blob))

        def walk(parent):
            parts = []
            for block in parent.iter_inner_content():
                if isinstance(block, Paragraph):
                    parts.append(block.text)
                elif isinstance(block, Table):
                    seen = set()
                    rows = [list(row.cells) for row in block.rows]
                    # Some templates use a table as a two-column page layout.
                    columns = [[row[i] for row in rows if len(row)>i] for i in range(len(block.columns))]
                    independent = len(columns)==2 and all(any(heading(line.strip())
                        for cell in col for line in cell.text.splitlines()) for col in columns)
                    for cells in (columns if independent else rows):
                        for cell in cells:
                            if cell._tc not in seen:
                                seen.add(cell._tc)
                                parts.extend(walk(cell))
            return parts

        parts = walk(doc)
        # Text boxes are not surfaced as normal python-docx paragraphs.
        boxes = doc.element.xpath('.//w:txbxContent')
        for box in boxes:
            ns = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
            for para in box.iter(ns+'p'):
                value = ''.join(t.text or '' for t in para.iter(ns+'t'))
                if value and value not in parts:
                    parts.append(value)
        if boxes:
            warnings.append('检测到Word文本框，已追加提取，请核对其所属章节。')
        return '\n'.join(parts), warnings
    if ext in ('.txt', '.md'):
        for encoding in ('utf-8-sig', 'utf-16' if blob[:2] in (b'\xff\xfe', b'\xfe\xff') else 'gb18030'):
            try:
                return blob.decode(encoding), warnings
            except UnicodeError:
                pass
        raise ResumeError('文本编码无法识别，请另存为UTF-8')
    raise ResumeError('支持 PDF、DOCX、TXT、MD；旧版DOC请先另存为DOCX')


def infer_kind(line):
    if re.search(r'实习生|实习岗位|\bintern\b', line, re.I):
        return '实习经历'
    if re.search(r'(大学|学院|university|college).*(本科|硕士|博士|学士|专科|bachelor|master|ph\.?d)', line, re.I):
        return '学历'
    if re.match(r'^(?:熟悉|熟练|掌握|精通|了解|擅长)', line):
        return '工作技能'
    if re.match(r'^(?:项目名称|项目名|项目\s*[一二三四五六七八九\d]+)\s*[:：、.]', line):
        return '项目'
    return '其他'


def parse_sections(raw_text, warnings=None):
    text = normalize(raw_text)
    if len(text) > 60000:
        raise ResumeError('简历超过60000字，请精简后上传；系统不会截断原文')
    if len(re.sub(r'\s', '', text)) < 10:
        raise ResumeError('未提取到足够的文字。扫描PDF需要先OCR，或直接粘贴简历正文')
    warnings = list(warnings or [])
    cards, current, explicit = [], '其他', False
    lines, method, section_no = [], 'unclassified', 0
    page, source_line, blank = 1, 0, False

    def flush():
        nonlocal lines
        if not lines:
            return
        # Preserve full entries; chunks above the per-card API limit are split
        # without discarding any characters or source references.
        chunks, chunk, length = [], [], 0
        for record in lines:
            for offset in range(0, len(record['text']), 10000):
                part = {**record, 'text': record['text'][offset:offset+10000]}
                if length + len(part['text']) + 1 > 12000 and chunk:
                    chunks.append(chunk); chunk=[]; length=0
                chunk.append(part); length += len(part['text']) + 1
        if chunk:
            chunks.append(chunk)
        for chunk in chunks:
            value = '\n'.join(x['text'] for x in chunk)
            cards.append({'id':f'e{len(cards)+1}', 'kind':current, 'title':chunk[0]['text'][:100],
                          'text':value, 'confirmed':False, 'method':method,
                          'source':{'pages':sorted({x['page'] for x in chunk}),
                                    'lines':[x['line'] for x in chunk]}, 'section':section_no})
        lines=[]

    for physical in text.splitlines(keepends=True):
        # splitlines includes form-feed boundaries; retain PDF page locations.
        source_line += 1
        for line in INLINE.split(physical.strip()):
            line = line.strip()
            if not line:
                blank = True
                continue
            found = heading(line)
            if found:
                flush(); current, line = found; method='heading'; explicit=True; section_no+=1
                blank=False
                if not line:
                    continue
            elif not explicit:
                guess = infer_kind(line)
                if guess != '其他' and guess != current:
                    flush(); current=guess; method='inferred'
            if lines:
                item_start = bool(re.match(r'^(项目名称|项目名|公司名称|实习单位|学校名称)\s*[:：]', line))
                dated = bool(DATE_RANGE.search(line)) and len(line) < 180 and not re.match(r'^[•●\-]|^(?:负责|参与|完成|实现|提升|结果)', line)
                new_date = dated and any(DATE_RANGE.search(x['text']) for x in lines)
                title = None
                if new_date:
                    previous = lines[-1]['text']
                    if (len(previous)<80 and not DATE_RANGE.search(previous)
                        and not re.search(r'[:：。.!！;；]', previous)
                        and not re.match(r'^[•●\-]|^(?:负责|参与|完成|实现|提升|使用|开发|设计|优化|主修|熟悉|熟练|掌握)',previous)):
                        title = lines.pop()
                if item_start or new_date or (blank and current == '其他'):
                    flush()
                    if title:
                        lines.append(title)
            lines.append({'text':line, 'page':page, 'line':source_line})
            blank=False
        if '\f' in physical:
            page += physical.count('\f')
    flush()
    sections = {kind:[c for c in cards if c['kind']==kind] for kind in KINDS}
    if any(c['method']=='inferred' for c in cards):
        warnings.append('部分内容缺少章节标题，按文本线索暂分类，请核对。')
    if sections['其他']:
        warnings.append('其他/待归类内容已保留，可手动移动分类；不会静默丢弃。')
    if not any(c['method']=='heading' for c in cards):
        warnings.append('没有识别到标准章节标题；可在原文中添加“项目经历、实习经历、教育背景、专业技能”后重新解析。')
    return {'text':text, 'cards':cards, 'sections':sections, 'warnings':warnings,
            'parser_version':'sections-v2', 'method':'local-structure',
            'summary':{k:len(v) for k,v in sections.items()}}
