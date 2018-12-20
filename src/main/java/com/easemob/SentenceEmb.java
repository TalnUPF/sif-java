package com.easemob;

import org.apache.commons.math3.linear.*;
import java.util.List;

public class SentenceEmb
{
	private final Vectors vectors;
	private final Weights weights;
	private final int num_dimensions;
	private final double default_weight;

	public SentenceEmb(Vectors vectors, Weights weights)
	{
		this.vectors = vectors;
		this.weights = weights;
		num_dimensions = vectors.getNumDimensions();
		default_weight = weights.getMinimumWeight();
	}

	/* convert a list of words to a weighted vector, return [1 x wordVecLen] */
	public RealVector weightedAvg(List<String> text)
	{
		int sentLen = text.size();
		RealMatrix emb = new Array2DRowRealMatrix(sentLen, num_dimensions);
		RealVector w = new ArrayRealVector(sentLen);
		for (int i = 0; i < sentLen; i++)
		{
			String word = text.get(i);
			double weight = weights.get_weight(word).orElse(default_weight);

			w.setEntry(i, weight);
			final double[] array = vectors.get(word).orElse(new double[num_dimensions]);
			final Array2DRowRealMatrix vector = new Array2DRowRealMatrix(array);
			emb.setRowMatrix(i, vector.transpose());
		}

		return emb.preMultiply(w).mapMultiply(1.0 / w.getDimension());
	}

	/**
	 * Convert a list of words to weighted vector and remove
	 * the most common shared principle component(s).
	 *
	 * @param text tokenized sentence made of token strings
	 * @param k    remove how many principle components? (default 0)
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

	public RealMatrix embedding(List<String> text)
	{
		return embedding(text, 0);
	}

	/**
	 * Convert a list of sentences to weighted vectors and remove
	 * the most common shared principle component(s).
	 *
	 * @param texts tokenized sentences made of token strings
	 * @param k     remove how many principle components? (default 1)
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

	public RealMatrix matrixEmbedding(List<List<String>> texts)
	{
		return matrixEmbedding(texts, 1);
	}

	/* remove principle components */
	private RealMatrix removePrincipleComponents(RealMatrix m, int k)
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
