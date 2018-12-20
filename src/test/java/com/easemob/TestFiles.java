package com.easemob;

import java.nio.file.Path;
import java.nio.file.Paths;

public class TestFiles
{
	public final static Path test_vectors_file = Paths.get(System.getProperty("user.dir") + "/data/glove.test.txt");
	public final static Path test_weights_file = Paths.get(System.getProperty("user.dir") + "/data/idf.txt");
}
