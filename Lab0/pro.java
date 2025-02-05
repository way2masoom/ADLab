import java.util.Scanner;

// Base class
class Plate {
    double length, width;

    // Constructor to initialize length and width
    Plate(double length, double width) {
        this.length = length;
        this.width = width;
        System.out.println("Plate Constructor: Length = " + length + ", Width = " + width);
    }
}

// Derived class from Plate
class Box extends Plate {
    double height;

    // Constructor to initialize length, width, and height
    Box(double length, double width, double height) {
        super(length, width); // Calling Plate constructor
        this.height = height;
        System.out.println("Box Constructor: Height = " + height);
    }
}

// Derived class from Box
class WoodBox extends Box {
    double thickness;

    // Constructor to initialize length, width, height, and thickness
    WoodBox(double length, double width, double height, double thickness) {
        super(length, width, height); // Calling Box constructor
        this.thickness = thickness;
        System.out.println("WoodBox Constructor: Thickness = " + thickness);
    }
}

// Main class to test the execution
public class MultiLevelInheritanceDemo {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);

        // Taking input from the user
        System.out.print("Enter length: ");
        double length = scanner.nextDouble();
        System.out.print("Enter width: ");
        double width = scanner.nextDouble();
        System.out.print("Enter height: ");
        double height = scanner.nextDouble();
        System.out.print("Enter thickness: ");
        double thickness = scanner.nextDouble();

        // Creating an instance of WoodBox
        WoodBox woodBox = new WoodBox(length, width, height, thickness);

        scanner.close();
    }
}

//Output
Enter length: 10
Enter width: 5
Enter height: 8
Enter thickness: 2
Plate Constructor: Length = 10.0, Width = 5.0
Box Constructor: Height = 8.0
WoodBox Constructor: Thickness = 2.0
