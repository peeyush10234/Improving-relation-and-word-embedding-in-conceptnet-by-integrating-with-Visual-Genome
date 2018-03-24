import json
import divisi2
import numpy as np
from csc.divisi2.blending import blend
from csc.divisi.flavors import ConceptByFeatureMatrix

annotations = json.load(open('/home/salvation/Desktop/annotations_test.json'))
objects = json.load(open('/home/salvation/Desktop/objects.json'))
predicates = json.load(open('/home/salvation/Desktop/predicates.json'))

imgs = list()
for keys in annotations:
	imgs.append(keys)


p = np.load('/home/salvation/visual_genome_relations_1.npy')
q = np.load('/home/salvation/vg_relationships/visual_genome_relations_3.npy')


# For divisi

# weighted_triples = list()
# for img in imgs:
# 	for annotation in annotations[img]:
# 		relation = predicates[annotation['predicate']]
# 		obj = objects[annotation['object']['category']]
# 		subject = objects[annotation['subject']['category']]
# 		rel_triple = (obj, relation, subject)
# 		weight = 4.0
# 		weighted_triple = (rel_triple, weight)
# 		weighted_triples.append(weighted_triple)

# matrix = ConceptByFeatureMatrix.from_triples(weighted_triples)
# normalized_matrix = matrix.normalized()
# print(type(normalized_matrix))

# For divisi2

weighted_relations = list()
# for img in imgs:
# 	for annotation in annotations[img]:
# 		# relation = predicates[annotation['predicate']]
# 		obj = objects[annotation['object']['category']]
# 		subject = objects[annotation['subject']['category']]
# 		weight = 4.0
#    	rel_triple = (weight, obj, subject)
# 		# weighted_triple = (rel_triple, weight)
# 		weighted_relations.append(rel_triple)
obj_list = []
for idx in range(len(p)):
	obj = p[idx][2]
	subj = p[idx][0]
	weight = 4.0
	rel_triple = (weight, obj, subj)
	obj_list.append(obj)
 		# weighted_triple = (rel_triple, weight)
	weighted_relations.append(rel_triple)

for idx in range(len(q)):
	obj = q[idx][2]
	subj = q[idx][0]
	weight = 4.0
	rel_triple = (weight, obj, subj)
	obj_list.append(obj)
 		# weighted_triple = (rel_triple, weight)
	weighted_relations.append(rel_triple)

print len(weighted_relations)	
#print len(obj_list)
obj_list = set(obj_list)	
print len(obj_list)
matrix = divisi2.make_sparse(weighted_relations)
#print matrix

# ConceptNet Matrix
A = divisi2.network.conceptnet_matrix('en')
A_concept_axes, A_axis_weights, A_feature_axes = A.svd(k=100)

blended_matrix = blend([matrix, A])
concept_axes, axis_weights, feature_axes = blended_matrix.svd(k=100)

common_objects = list(set(obj_list).intersection(A.row_labels))
print len(A.row_labels)

# Save embeddings for ConceptNet
cnet_object_embeddings = np.array(
	[A_concept_axes.row_named(obj) for obj in common_objects])
np.save('cnet_object_embeddings.npy', cnet_object_embeddings)

# Save embeddings for Blended Matrix
blended_object_embeddings = np.array(
	[concept_axes.row_named(obj) for obj in common_objects])
np.save('blended_object_embeddings.npy', blended_object_embeddings)
v = []
for idx in obj_list:
	#print idx
	v.append(concept_axes.row_named(i	
	