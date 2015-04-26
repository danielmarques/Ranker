package br.puc.drm.ranker.tests;

import static org.junit.Assert.*;

import org.junit.Test;

import br.puc.drm.ranker.experiments.RankExp;

public class RankExpTest {

	@Test
	public void test() {
		
		String[] args = new String[1];
		args[0] = "/home/daniel/workspace/RankerExp";
		
		RankExp exp = new RankExp();
		
		exp.main(args);
	}
}
