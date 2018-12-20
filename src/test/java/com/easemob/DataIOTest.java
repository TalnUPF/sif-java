package com.easemob;

import org.junit.Test;

import java.nio.file.Paths;

import static com.easemob.TestFiles.test_weights_file;

public class DataIOTest
{
	public DataIOTest() {}

	@Test
    public void testLoadingFiles() throws Exception
	{
		DataIO data = new DataIO(TestFiles.test_vectors_file, test_weights_file);
        assert data.getNumDimensions() > 0;
    }
}