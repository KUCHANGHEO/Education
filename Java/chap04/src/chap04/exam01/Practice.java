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
			System.out.println("1입니다");
			break;
		case 2:
			System.out.println("2입니다");
			break;
		case 3:
			System.out.println("3입니다");
			break;
		default:
			System.out.println("4입니다");
			break;
		}
		
		int[][] ary = new int[2][3];
		System.out.println(ary.length);
		
		String season[] = {"봄","여름","가을","겨울"};
		
		for (int i = 0; i < season.length; i++) {
			System.out.println("우리나라 계절은"+ season[i] + "로 구성되어 있습니다.");
		}
		
		
	}

}

