package com.easemob;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

public class DataIO
{
	private final Map<String, double[]> vectors = new HashMap<>();
	private final Map<String, Double> weights = new HashMap<>();
	private final double min_weight;

	public DataIO(Path words_file, Path vectors_file) throws IOException
	{
		try(BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(vectors_file.toFile()))))
		{
			String line;
			while ((line = br.readLine()) != null)
			{
				String[] tmp = line.split(" ");
				double[] vecArr = new double[tmp.length - 1];
				for (int i = 1; i < tmp.length; i++)
				{
					vecArr[i - 1] = Double.valueOf(tmp[i]);
				}
				vectors.put(tmp[0], vecArr);
			}
		}

		try(BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(words_file.toFile()))))
		{
			String line;
			while ((line = br.readLine()) != null)
			{
				String[] tmp = line.split("\\s");
				weights.put(tmp[0], Double.valueOf(tmp[1]));
			}

		}

		min_weight = weights.values().stream()
				.mapToDouble(d -> d)
				.min().orElse(0.0);
	}

	public double get_weight(String item)
	{
		return weights.getOrDefault(item, min_weight);
	}

	public double[] get_vector(String item)
	{
		return vectors.get(item);
	}

	public int getNumDimensions()
	{
		return vectors.values().iterator().next().length;
	}
}