package chap04.exam01;

public class Practice {

	public static void main(String[] args) {
		int sum = 0;
		for (int i = 1; i < 101; i++) {
			if (i % 5 == 0) {
				sum += i;
			} else {
				continue;
			}
		}
		System.out.println(sum);
		
		int num = (int)(Math.random()*4) + 1;
		
		switch (num) {
		case 1:
			System.out.println("1�Դϴ�");
			break;
		case 2:
			System.out.println("2�Դϴ�");
			break;
		case 3:
			System.out.println("3�Դϴ�");
			break;
		default:
			System.out.println("4�Դϴ�");
			break;
		}
		
		int[][] ary = new int[2][3];
		System.out.println(ary.length);
		
		String season[] = {"��","����","����","�ܿ�"};
		
		for (int i = 0; i < season.length; i++) {
			System.out.println("�츮���� ������"+ season[i] + "�� �����Ǿ� �ֽ��ϴ�.");
		}
		
		
	}

}

