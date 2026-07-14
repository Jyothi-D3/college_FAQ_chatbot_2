import warnings; warnings.filterwarnings("ignore")
from governance import _ask, _load_store, _has_citation

store = _load_store()
ans, docs, _ = _ask(store, "What accreditations does BVRIT have?")
print("Answer:", ans[:500])
print("Has citation:", _has_citation(ans))
print("Docs retrieved:", len(docs))
