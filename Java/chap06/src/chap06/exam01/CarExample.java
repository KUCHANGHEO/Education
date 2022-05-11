package chap06.exam01;

public class CarExample {

	public static void main(String[] args) {
//		Car myCar = new Car(100);
	
/*		Car myCar = new Car("white", 100);
		
		System.out.println(myCar.color);
		
		System.out.println(myCar.Speed);*/

		// 모델을 "GV80" 으로 myCar를 만들어 봅시다
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
			System.out.println("출발합니다.");
			myCar.run();
		}
		
		if(myCar.isLeftGas()) {
			System.out.println("gas를 주입할 필요가 없습니다");
		} else {
			System.out.println("gas가 0입니다.");
		}
		
	}

}
