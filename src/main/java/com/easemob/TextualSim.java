package com.easemob;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.function.Function;

public class TextualSim
{
	private final SentenceEmb sentenceEmb;

	public TextualSim(Function<String, Optional<double[]>> vectors, int num_dimensions,
	                  Function<String, Double> weights)
	{
		this.sentenceEmb = new SentenceEmb(vectors, num_dimensions, weights);
	}

	public RealVector getWeightedAverage(List<String> tokens) { return sentenceEmb.weightedAvg(tokens); }
	public RealMatrix getEmbedding(List<String> tokens) { return sentenceEmb.embedding(tokens, 1); }
	public RealMatrix getEmbedding(List<String> tokens, int k) { return sentenceEmb.embedding(tokens, k); }

	/* evaluate textual similarity between 2 sentences */
	public double score(List<String> t1, List<String> t2) { return score(t1, t2, 1); }
	public double score(List<String> t1, List<String> t2, int k)
	{
		RealMatrix e1 = sentenceEmb.embedding(t1, k);
		RealMatrix e2 = sentenceEmb.embedding(t2, k);
		// calculate cosine angle
		RealMatrix inn = inner(e1, e2);
		RealMatrix e1Norm = sqrt(inner(e1, e1));
		RealMatrix e2Norm = sqrt(inner(e2, e2));
		return div(div(inn, e1Norm), e2Norm).getEntry(0, 0);
	}

	/* evaluate similarity between 2 vectors resulting from the weighted average of word embeddings of tokens in a sentence */
	public double score(double [] v1, double [] v2) { return score(v1, v2, 1); }
	public double score(double [] v1, double [] v2, int k)
	{
		// calculate cosine angle
		RealMatrix e1 = sentenceEmb.embedding(new ArrayRealVector(v1, false), k);
		RealMatrix e2 = sentenceEmb.embedding(new ArrayRealVector(v2, false), k);
		RealMatrix inn = inner(e1, e2);
		RealMatrix e1Norm = sqrt(inner(e1, e1));
		RealMatrix e2Norm = sqrt(inner(e2, e2));
		return div(div(inn, e1Norm), e2Norm).getEntry(0, 0);
	}

	/* [m x s], [m x s] -> [1, m] */
	private RealMatrix inner(RealMatrix m1, RealMatrix m2)
	{
		if (m1.getColumnDimension() != m2.getColumnDimension() || m1.getRowDimension() != m2.getRowDimension())
		{
			throw new RuntimeException("m1, m2 must be the same shape");
		}
		int m = m1.getRowDimension();
		int n = m2.getColumnDimension();
		// multiply m1, m2 element by element
		RealMatrix tmp = new Array2DRowRealMatrix(m, n);
		for (int i = 0; i < m; i++)
		{
			for (int j = 0; j < n; j++)
			{
				double x = m1.getEntry(i, j) * m2.getEntry(i, j);
				tmp.setEntry(i, j, x);
			}
		}
		// sum over along axis 1
		RealMatrix res = new Array2DRowRealMatrix(1, m);
		for (int i = 0; i < m; ++i)
		{
			double sum = Arrays.stream(tmp.getRowVector(i).toArray()).sum();
			res.setEntry(0, i, sum);
		}
		return res;
	}

	/* sqrt m by element */
	private RealMatrix sqrt(RealMatrix m)
	{
		for (int i = 0; i < m.getRowDimension(); ++i)
		{
			for (int j = 0; j < m.getColumnDimension(); ++j)
			{
				m.setEntry(i, j, Math.sqrt(m.getEntry(i, j)));
			}
		}
		return m;
	}

	/* divide m1 by element in m2 accordingly, e.g. [3,6,9] / [3,3,3] -> [1,2,3] */
	private RealMatrix div(RealMatrix m1, RealMatrix m2)
	{
		int r1 = m1.getRowDimension();
		int r2 = m2.getRowDimension();
		int c1 = m1.getColumnDimension();
		int c2 = m2.getColumnDimension();
		if (r1 != r2 || c1 != c2)
		{
			throw new RuntimeException("m1, m2 must be the same shape ([" + r1 + " x " + c1 + "], [" + r2 + " x " + c2 + "])");
		}
		RealMatrix res = new Array2DRowRealMatrix(r1, c1);
		for (int i = 0; i < r1; i++)
		{
			for (int j = 0; j < c1; j++)
			{
				double x = m1.getEntry(i, j) / m2.getEntry(i, j);
				res.setEntry(i, j, x);
			}
		}
		return res;
	}
}