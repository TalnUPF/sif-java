package com.easemob;

import java.util.OptionalDouble;

public interface Weights
{
	OptionalDouble get_weight(String item);
	double getMinimumWeight();
}
