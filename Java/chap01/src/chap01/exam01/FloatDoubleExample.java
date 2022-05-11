package chap01.exam01;

public class FloatDoubleExample {

	public static void main(String[] args) {
		double var1 = 3.14;
		//float var2 = 3.14; 컴파일 에러
		float var3 = 3.14F;
		System.out.println(var1);
		System.out.println(var3);
		
		double var4 = 1.234456789;
		System.out.println(var4);
		
		float var5 = 1.234456789F;
		System.out.println(var5);
		
		//e 사용하기
		
		int var6 = 123000;
		System.out.println(var6);
		
		double var7 = 123e5;
		System.out.println(var6);

		float var8 = 123e-3F;
		System.out.println(var6);
		
		
	}

}
