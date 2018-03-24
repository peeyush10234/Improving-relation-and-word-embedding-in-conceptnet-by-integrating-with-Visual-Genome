# Improving-relation-and-word-embedding-in-conceptnet-by-integrating-with-Visual-Genome

Modern NLP applications broadly used potent tools like word embeddings and others like dependency parsing, which can be based on distributional hypothesis or on structural knowledge. All these semantic based embeddings somewhere lacks in determining the visual content of the word or entity. Recent works are able to find out the embeddings for visual words but this approach can’t be used to generate the embeddings based on the visual relations. Our model first find the visual relations through Visual Genome and VRD Dataset between entities and than align and integrates these relations with ConceptNet. ConceptNet is a knowledge graph that connects words and phrases of natural language with labeled edges. Its knowledge is collected from many sources like DBpedia, Cyc and Wordnet. Conceptnet Numberbatch is the hybrid model which learns its word embeddings from ConceptNet as well as from Distributional Semantics.
