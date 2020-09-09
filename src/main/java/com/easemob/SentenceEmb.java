package com.easemob;

import org.apache.commons.math3.linear.*;

import java.util.List;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;
import java.util.stream.IntStream;

public class SentenceEmb
{
	private final Function<String, Optional<double[]>> vectors;
	private final int num_dimensions;
	private final Function<String, Double> weights;

	public SentenceEmb(Function<String, Optional<double[]>> vectors, int num_dimensions,
	                   Function<String, Double> weights)
	{
		this.vectors = vectors;
		this.weights = weights;
		this.num_dimensions = num_dimensions;
	}

	public RealMatrix embedding(List<String> text)
	{
		return embedding(text, 1);
	}

	/**
	 * Convert a list of words to weighted vector and remove
	 * the most common shared principle component(s).
	 *
	 * @param text tokenized sentence made of token strings
	 * @param k    remove how many principle components?
	 * @return embedded sentence using weights and word vector
	 */
	public RealMatrix embedding(List<String> text, int k)
	{
		RealVector m = weightedAvg(text);
		RealMatrix res = new Array2DRowRealMatrix(1, m.getDimension());
		res.setRowVector(0, m);
		if (k > 0)
		{
			res = removePrincipleComponents(res, k);
		}
		return res;
	}

	/**
	 * Convert a list of words to weighted vector and remove
	 * the most common shared principle component(s).
	 *
	 * @param w vector resulting from weighted average of a sequence of tokens
	 * @param k    remove how many principle components?
	 * @return embedded sentence using weights and word vector
	 */
	public RealMatrix embedding(RealVector w, int k)
	{
		RealMatrix res = new Array2DRowRealMatrix(1, w.getDimension());
		res.setRowVector(0, w);
		if (k > 0)
		{
			res = removePrincipleComponents(res, k);
		}
		return res;
	}

	public RealMatrix matrixEmbedding(List<List<String>> texts)
	{
		return matrixEmbedding(texts, 1);
	}

	/**
	 * Convert a list of sentences to weighted vectors and remove
	 * the most common shared principle component(s).
	 *
	 * @param texts tokenized sentences made of token strings
	 * @param k     remove how many principle components?
	 * @return embedded sentences using weights and word vector
	 */
	public RealMatrix matrixEmbedding(List<List<String>> texts, int k)
	{
		RealMatrix res = new Array2DRowRealMatrix(texts.size(), num_dimensions);
		for (int i = 0; i < texts.size(); ++i)
		{
			List<String> text = texts.get(i);
			res.setRowMatrix(i, embedding(text, 0));
		}

		if (k > 0)
			res = removePrincipleComponents(res, k);

		return res;
	}

	/* convert a list of words to a weighted vector, return [1 x wordVecLen] */
	public RealVector weightedAvg(List<String> text)
	{
		AtomicInteger num_vectors = new AtomicInteger();
		final double[] sentenceVector = new double[num_dimensions];

		for (String word : text)
		{
			final Optional<double[]> array = vectors.apply(word);
			if (array.isPresent())
			{
				num_vectors.incrementAndGet();
				double[] v = array.get();
				double weight = weights.apply(word);
				IntStream.range(0, num_dimensions).forEach(i -> sentenceVector[i] += v[i] * 0.001 / (0.001 + weight));
			}
		}

		IntStream.range(0, num_dimensions).forEach(i -> sentenceVector[i] /= (1.0 / num_vectors.get()));
		return new ArrayRealVector(sentenceVector);
	}

	/* remove principle components */
	public RealMatrix removePrincipleComponents(RealMatrix m, int k)
	{
		RealMatrix pc = getTruncatedSVD(m, k);
		return m.subtract(m.multiply(pc.transpose()).multiply(pc));
	}

	/* calculate principle components */
	private RealMatrix getTruncatedSVD(RealMatrix m, int k)
	{
		SingularValueDecomposition svd = new SingularValueDecomposition(m);

		double[][] truncatedU = new double[svd.getU().getRowDimension()][k];
		double[][] truncatedS = new double[k][k];
		double[][] truncatedVT = new double[k][svd.getVT().getColumnDimension()];

		svd.getU().copySubMatrix(0, truncatedU.length - 1, 0, k - 1, truncatedU);
		svd.getS().copySubMatrix(0, k - 1, 0, k - 1, truncatedS);
		svd.getVT().copySubMatrix(0, k - 1, 0, truncatedVT[0].length - 1, truncatedVT);

		RealMatrix u = new Array2DRowRealMatrix(truncatedU);
		RealMatrix s = new Array2DRowRealMatrix(truncatedS);
		RealMatrix vt = new Array2DRowRealMatrix(truncatedVT);

		return u.multiply(s).multiply(vt);
	}

}
