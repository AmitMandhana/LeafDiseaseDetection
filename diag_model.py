from sentence_transformers import SentenceTransformer
import inspect

m = SentenceTransformer('model')
print('SentenceTransformer object:', type(m))

# print top-level modules
try:
    modules = getattr(m, '_modules', None)
    if modules:
        print('Number of modules:', len(modules))
        for i, mod in enumerate(modules):
            print(f'  Module {i}:', type(mod), getattr(mod, '__class__', None))
            # print some attributes if available
            for attr in ('auto_model', 'auto_model_name', 'vector_size', 'pooling'): 
                if hasattr(mod, attr):
                    print(f'    has attribute {attr}:', getattr(mod, attr))
    else:
        # fallback: inspect attributes
        for name in dir(m):
            if not name.startswith('_'):
                print(name)
except Exception as e:
    print('Error inspecting modules:', e)

# try to print model_name_or_path if present
try:
    print('model_name_or_path:', getattr(m, 'model_name_or_path', None))
except Exception:
    pass

# encode a short sentence to ensure model works
s = 'test sentence for embeddings'
emb = m.encode(s)
print('Embedding length:', len(emb))
print('Done')
