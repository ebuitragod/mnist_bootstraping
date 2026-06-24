MNIST bootstrapping

This task is about quick automatic pre-labelling of MNIST dataset (its training part) using only some seed data from its test subset.

1. Download the MNIST dataset
2. Use the test split as the seed set, pretend we have no labels for the training subset.
3. Train an initial classifier model on the seed set.
4. Using the initial model, generate embeddings for the training subset.
5. Project the embeddings onto 2D space, visualize them and automatically cluster them into 10 clusters.
6. Determine the classes for the resulting clusters, use them as the initial labels for training samples.

From now on you can use the training set to re-train your model, but only with the labels you created, not the groundtruth ones.

7. Find the errors in the initial labels in unsupervised fashion, fix them automatically (as much as possible) to create the final set of labels.
8. Calculate the pre-labelling accuracy before and after correction using the groundtruth as the reference.

Meta-instructions:
 - Wrap your solution as a dockerized Python project. Share it as the source code, don't send the generated Docker image or any bulky files.
 - If you use coding agents, add agent instructions or skills the project too.
 - Think about the project structure, packaging and deployment.
 - Documentation is good.
