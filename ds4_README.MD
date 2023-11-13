The sentiment classification is based on LLM-based embedding & pair-wise text similarity score.
The package used is LlamaIndex: https://gpt-index.readthedocs.io/en/latest/examples/embeddings/huggingface.html

For each comment, it is embedded by 384-dimensional vector (L2-normalize).
For 5-category sentiment class, the category name is converted to a sentence, based on the template, e.g.
  The sentiment of comment is {sentiment-category}.

After embedding is ready, calculate matrix multiplication and the category having maximal score is predicted sentiment.

The other large Llama model may have good performance. But it cannot run in cpu labtop with few memory.
