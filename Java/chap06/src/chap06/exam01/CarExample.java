package chap06.exam01;

public class CarExample {

	public static void main(String[] args) {
//		Car myCar = new Car(100);
	
/*		Car myCar = new Car("white", 100);
		
		System.out.println(myCar.color);
		
		System.out.println(myCar.Speed);*/

		// ���� "GV80" ���� myCar�� ����� ���ô�
		/*Car myCar = new Car("GV80","white", 100);
		
		System.out.println(myCar.model);
		System.out.println(myCar.color);
		System.out.println(myCar.speed);
		System.out.println(myCar.company);
		System.out.println(Car.company);*/
		
		Car myCar = new Car(100);
		
		myCar.setGas(5);
		
		boolean gasState = myCar.isLeftGas();
		
		if(gasState) {
			System.out.println("����մϴ�.");
			myCar.run();
		}
		
		if(myCar.isLeftGas()) {
			System.out.println("gas�� ������ �ʿ䰡 �����ϴ�");
		} else {
			System.out.println("gas�� 0�Դϴ�.");
		}
		
	}

}
