package chap01.exam02;

public class StringEqualExample {

	public static void main(String[] args) {
		String strVar1 = "�߾�";
		String strVar2 = "�߾�";
		String strVar3 = new String("�߾�");
		
		System.out.println(strVar1 == strVar2);
		System.out.println(strVar1 == strVar3);
		System.out.println();
		System.out.println(strVar1.equals(strVar2));//equals()���ڸ� ��
		System.out.println(strVar1.equals(strVar3));

	}

}



