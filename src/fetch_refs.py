import requests,xml.etree.ElementTree as ET,json,re,time
queries=[
 'perturb-seq single-cell CRISPR', 'single-cell perturbation prediction machine learning', 'single-cell gene expression perturbation model',
 'pathway neural network transcriptomics', 'single-cell RNA-seq deep learning', 'Reactome pathway database',
 'Nature Communications single-cell perturbation', 'cellular perturbation foundation model', 'single-cell causal inference perturbation',
 'gene regulatory network perturbation single-cell', 'single-cell transcriptomic representation learning', 'explainable AI genomics'
]
ids=[]
for q in queries:
 j=requests.get('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi',params={'db':'pubmed','term':q,'retmax':12,'retmode':'json','sort':'relevance'},timeout=30).json(); ids += j['esearchresult']['idlist']
ids=list(dict.fromkeys(ids))[:90]
xml=requests.get('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi',params={'db':'pubmed','id':','.join(ids),'retmode':'xml'},timeout=60).text
root=ET.fromstring(xml); refs=[]
def txt(el,path):
 x=el.find(path); return ''.join(x.itertext()).strip() if x is not None else ''
def esc(s): return s.replace('{','\\{').replace('}','\\}')
for a in root.findall('.//PubmedArticle'):
 pmid=txt(a,'.//PMID'); title=txt(a,'.//ArticleTitle'); title=re.sub(r'\s+',' ',title); journal=txt(a,'.//Journal/Title'); year=txt(a,'.//PubDate/Year') or txt(a,'.//PubDate/MedlineDate')[:4]; year=re.sub(r'[^0-9].*','',year) or '2020'; vol=txt(a,'.//JournalIssue/Volume'); issue=txt(a,'.//JournalIssue/Issue'); pages=txt(a,'.//Pagination/MedlinePgn'); doi=''
 for aid in a.findall('.//ArticleId'):
  if aid.attrib.get('IdType')=='doi': doi=aid.text or ''
 authors=[]
 for au in a.findall('.//AuthorList/Author'):
  ln=txt(au,'LastName'); ini=txt(au,'Initials');
  if ln: authors.append(f'{ln} {ini}')
 if not title or not authors: continue
 key='pmid'+pmid
 refs.append({'key':key,'pmid':pmid,'title':title,'journal':journal,'year':year,'volume':vol,'issue':issue,'pages':pages,'doi':doi,'authors':authors})
# remove duplicate titles and keep 60 direct refs
seen=set(); out=[]
for r in refs:
 k=re.sub(r'[^a-z0-9]','',r['title'].lower())
 if k in seen: continue
 seen.add(k); out.append(r)
 if len(out)>=70: break
lines=[]
for r in out:
 auth=' and '.join(r['authors'][:12]); bib=f"@article{{{r['key']},\n  author = {{{auth}}},\n  title = {{{esc(r['title'])}}},\n  journal = {{{esc(r['journal'])}}},\n  year = {{{r['year']}}},\n  volume = {{{r['volume']}}},\n  number = {{{r['issue']}}},\n  pages = {{{r['pages']}}},\n  doi = {{{r['doi']}}},\n  pmid = {{{r['pmid']}}}\n}}\n"; lines.append(bib)
open('paper/references.bib','w').write('\n'.join(lines)); open('literature/pubmed_refs.json','w').write(json.dumps(out,indent=2)); print(len(out))
