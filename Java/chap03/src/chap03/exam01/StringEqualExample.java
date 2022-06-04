package chap01.exam02;

public class StringEqualExample {

	public static void main(String[] args) {
		String strVar1 = "Áß¾Ó";
		String strVar2 = "Áß¾Ó";
		String strVar3 = new String("Áß¾Ó");
		
		System.out.println(strVar1 == strVar2);
		System.out.println(strVar1 == strVar3);
		System.out.println();
		System.out.println(strVar1.equals(strVar2));//equals()±ÛÀÚ¸¸ ºñ±³
		System.out.println(strVar1.equals(strVar3));

	}

}



