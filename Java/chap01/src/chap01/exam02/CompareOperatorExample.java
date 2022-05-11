package chap01.exam02;

public class CompareOperatorExample {

	public static void main(String[] args) {
		double v4 = 0.1;
		float v5 = 0.1f;
		System.out.println(v4 == v5);
		System.out.println((float)v4 == v5);
		System.out.println((int)(v4*10) == (int)(v5*10));
	}

}
/*
°á°ú°ª
false
true
true
*/