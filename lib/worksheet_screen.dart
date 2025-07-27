import 'package:flutter/material.dart';
import 'worksheet_prompt_screen.dart';

class WorksheetScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Worksheet Generator')),
      body: Center(
        child: Card(
          elevation: 4,
          child: ListTile(
            leading: Icon(Icons.playlist_add_check),
            title: Text('Start Generating Worksheet'),
            trailing: Icon(Icons.arrow_forward),
            onTap: () {
              Navigator.push(
                context,
                MaterialPageRoute(builder: (_) => WorksheetPromptScreen()),
              );
            },
          ),
        ),
      ),
    );
  }
}
