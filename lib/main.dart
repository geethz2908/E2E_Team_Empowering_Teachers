import 'package:flutter/material.dart';
import 'feature_screen.dart';

void main() {
  runApp(EduShayakApp());
}

class EduShayakApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'EduShayak',
      theme: ThemeData(
        primarySwatch: Colors.indigo,
      ),
      home: HomeScreen(),
    );
  }
}

class HomeScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('EduShayak Home'),
      ),
      body: Center(
        child: ElevatedButton(
          onPressed: () {
            Navigator.push(
              context,
              MaterialPageRoute(builder: (context) => FeatureScreen()),
            );
          },
          child: Text("Open EduShayak Tools"),
        ),
      ),
    );
  }
}
