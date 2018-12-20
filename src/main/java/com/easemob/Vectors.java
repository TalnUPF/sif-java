package com.easemob;

import java.util.Optional;

public interface Vectors
{
	Optional<double[]> get(String item);
	int getNumDimensions();
}
