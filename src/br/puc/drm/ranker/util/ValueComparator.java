package br.puc.drm.ranker.util;

import java.util.Comparator;
import java.util.Map;
import java.util.Set;

public class ValueComparator implements Comparator<Set<Integer>>{

	Map<Set<Integer>, Integer> base;
	
    public ValueComparator(Map<Set<Integer>, Integer> base) {
    	
        this.base = base;
    }

    // Note: this comparator imposes orderings that are inconsistent with equals.    
    public int compare(Set<Integer> a, Set<Integer> b) {
        if (base.get(a) >= base.get(b)) {
            return 1;
        } else {
            return -1;
        } // returning 0 would merge keys
    }
}
