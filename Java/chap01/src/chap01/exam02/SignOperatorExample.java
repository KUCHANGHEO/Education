package chap01.exam02;

public class SignOperatorExample {

	public static void main(String[] args) {
		int x = 10;
		int y = 10;
		int z = ++x + y ++;
		System.out.println("z: "+z);
		System.out.println("x: "+x);
		System.out.println("y: "+y);
		
		boolean play = true;
		System.out.println(play);
		
		play = !play;
		System.out.println(play);
		
		
		int v1 = 10;
		int v2 = ~v1;
		int v3 = ~v1 +1;
		System.out.println(v1);
		System.out.println(v2);
		System.out.println(v3);
		
		int v4 = -10;
		int v5 = ~v4;
		int v6 = ~v4 +1;
		System.out.println(v4);
		System.out.println(v5);
		System.out.println(v6);
	}

}
