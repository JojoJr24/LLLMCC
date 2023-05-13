// The module 'vscode' contains the VS Code extensibility API
// Import the module and reference it with the alias vscode in your code below
const vscode = require('vscode');
const axios = require('axios');

let url = "http://127.0.0.1:5000";

async function postData(url, data) {
	try {
		const response = await axios.post(url, data);
		return response.data.response;
	} catch (error) {
		// Handle error
		console.error(error);
	}
}

/**
 * @param {vscode.ExtensionContext} context
 */
function activate(context) {

	const editor = vscode.window.activeTextEditor;
	const folderName = vscode.workspace.workspaceFolders[0].uri.path.split("/").pop();
	const languageId = editor.document.languageId;

	let editURL = vscode.commands.registerCommand('lllmcc.setURL', () => {
		vscode.window.showInputBox({
			placeHolder: 'Enter LLM Service URL',
			prompt: 'http://127.0.0.1:5000'
		}).then((newValue) => {
			if (typeof newValue === 'string') {
				// Aquí puedes realizar las acciones que desees con el nuevo valor del campo de texto
				url = newValue;
				vscode.window.showInformationMessage('The new URL is: ' + newValue);
			}
		});
	});

	let run = vscode.commands.registerCommand('lllmcc.run', () => {
		const currentPosition = editor.selection.active;
		const codeAboveCursor = editor.document.getText(
			new vscode.Range(new vscode.Position(0, 0), currentPosition));

		if (editor) {
			async function callRun() {

				const response = await postData(url + "/addCode", { prompt: codeAboveCursor, lenguaje: languageId, proyecto: folderName });
				editor.edit((editBuilder) => {
					editBuilder.insert(currentPosition, response);
				})
			}
			callRun()
		}
	});

	let debug = vscode.commands.registerCommand('lllmcc.debug', () => {
		const editor = vscode.window.activeTextEditor;
		if (editor) {
			async function callDebug() {
				const selection = editor.selection;
				const selectedText = editor.document.getText(selection);
				const response = await postData(url + "/debugCode", { code: selectedText, lenguaje: languageId, proyecto: folderName});
				vscode.window.showInformationMessage(response);
			}
			callDebug()
		} else {
			vscode.window.showInformationMessage('No active text editor.');
		}
	});

	context.subscriptions.push(run);
	context.subscriptions.push(debug);
	context.subscriptions.push(editURL);

}



// This method is called when your extension is deactivated
function deactivate() { }

module.exports = {
	activate,
	deactivate
}
