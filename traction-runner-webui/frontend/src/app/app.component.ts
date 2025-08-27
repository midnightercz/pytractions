import { v4 as uuidv4 } from 'uuid';

import { Component, inject } from '@angular/core';
import { HttpClient, HttpClientModule } from '@angular/common/http';
import { RouterOutlet } from '@angular/router';
import { FormGroup, ReactiveFormsModule } from '@angular/forms';
import { FormlyForm, FormlyFormOptions, FormlyFieldConfig } from '@ngx-formly/core';
import { FormlyJsonschema } from '@ngx-formly/core/json-schema';
import { NgFor, NgIf } from '@angular/common';

import { catchError } from 'rxjs/operators';
import { throwError } from 'rxjs';

import { MatSidenavModule } from '@angular/material/sidenav';
import {MatSelectModule} from '@angular/material/select';
import {MatFormFieldModule} from '@angular/material/form-field';
import {ProgressSpinnerMode, MatProgressSpinnerModule} from '@angular/material/progress-spinner';
import {MatCardModule} from '@angular/material/card';
import {
  MatSnackBar,
  MatSnackBarAction,
  MatSnackBarActions,
  MatSnackBarLabel,
  MatSnackBarRef,
} from '@angular/material/snack-bar';
import {
  MatDialog,
  MatDialogActions,
  MatDialogClose,
  MatDialogContent,
  MatDialogRef,
  MatDialogTitle,
} from '@angular/material/dialog';
import {MatButtonModule} from '@angular/material/button';
import {MatTabsModule} from '@angular/material/tabs';
import {MatListModule} from '@angular/material/list';

import data from './test_data.json';

@Component({
  selector: 'loading-dialog',
  templateUrl: 'loading-dialog.html',
  imports: [MatDialogActions, MatDialogClose, MatDialogTitle, MatDialogContent, MatProgressSpinnerModule,
            MatCardModule],
})
export class LoadingDialog {
  readonly dialogRef = inject(MatDialogRef<LoadingDialog>);
}



@Component({
  selector: 'app-root',
  imports: [RouterOutlet, ReactiveFormsModule, FormlyForm, MatSidenavModule, MatButtonModule,
            HttpClientModule, MatSelectModule, MatFormFieldModule, MatTabsModule,
            MatCardModule, MatListModule, MatButtonModule,
            NgFor, NgIf],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css'
})
export class AppComponent {
  private _snackBar = inject(MatSnackBar);
  readonly loading_dialog = inject(MatDialog);

  form = new FormGroup({});
  title = 'view';
  model: any = {};
  fields: FormlyFieldConfig[];
  //data: any
  tractions_data: any
  traction_groups: string[] = []
  traction_modules: any = {}
  tractions: string[] = []
  selected_group: string = "";
  selected_traction: string = "";
  executed: boolean = false;
  selectedTabIndex: number = 0;
  logs: {level: string, log: string}[] = [];
  run_id: string = "";
  archives: {uid: string, user_desc: string}[] = [];
  logsLastId: number = 0;
  watching_interval_id: number = 0;
  info_bar_content: string = "";

  showLoading() {
    const dialogRef = this.loading_dialog.open(LoadingDialog, {
      width: '250px',
      height: '250px',
      disableClose: true,
      panelClass: 'transparent',
      data: { message: 'Loading...' },
    });
  }
  hideLoading() {
    this.loading_dialog.closeAll();
  }

  downloadModel() {
    const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(this.model, null, 2));
    const downloadAnchorNode = document.createElement('a');
    downloadAnchorNode.setAttribute('href', dataStr);
    downloadAnchorNode.setAttribute('download', 'state.json');
    document.body.appendChild(downloadAnchorNode); // required for Firefox
    downloadAnchorNode.click();
    downloadAnchorNode.remove();
  }

  uploadModel(event: any) {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e: any) => {
      try {
        this.model = JSON.parse(e.target.result);
        console.log('Uploaded JSON:', this.model);
      } catch (err) {
        console.error('Error parsing JSON:', err);
      }
    };
    reader.readAsText(file);
  }

  filter_tractions(group: string) {
    this.tractions = [];
    this.tractions_data.forEach((item: any) => {
      if (item.name === group) {
        (item.tractions as any[]).forEach((traction: any) => {
          this.tractions.push(traction.name);
          this.traction_modules[traction.name] = traction.module;
          this.selected_group = item.name;
        })
      }
    });
  }

  onSelectedGroup(group: string) {
    this.showLoading();
    this.filter_tractions(group);
    console.log("Selected " + group);
    this.hideLoading();
  }

  showError(error: string) {
    this._snackBar.open(error, 'Close');
  }

  supplyUserSchema(json: any) {
    json['properties']['user_description'] = {
      type: 'string',
      title: 'User Description',
      description: 'Provide a brief description of the model or its purpose.',
      default: '',
      maxLength: 500,
    };
  }

  buildForm(json: any) {
    this.supplyUserSchema(json);
    this.fields = [
      this.formlyJsonschema.toFieldConfig(json)
    ];
    this.fields.forEach((field: FormlyFieldConfig) => {
      if (field.validation == undefined) {
        field.validation = {show: true}
      }
      field.validation.show = true;
      field.expressionProperties = {'validation.show': 'model.showErrorState'}
    });
  }

  onSelectedTraction(traction: string) {
    this.showLoading();
    const module = this.traction_modules[traction];
    const selected_group = this.tractions_data.find((item: any) => item.name === this.selected_group);
    const traction_data = selected_group.tractions.find((item: any) => item.name === traction);
    this.http.get('/api/schema/' + module + '/' + traction, { responseType: 'json' })
    .pipe(
      catchError((error) => {
        this.showError('Error fetching schema: ' + error.message);
        console.error('Error fetching schema:', error);
        this.hideLoading();
        return throwError(() => new Error('Error fetching schema'));
      })
    )
    .subscribe(result=> {
      const json: any = result;
      this.buildForm(json);
      this.hideLoading();
      this.selected_traction = traction;
    });
  }

  fetch_archive() {
    fetch('/api/archive', {method: "GET"}).then(async (response) => {
      this.archives = await response.json();
      console.log("Fetched archived IDs:", this.archives);
    });
  }

  async selectArchived(id: string) {
    console.log("Selected archived ID:", id);
    this.showLoading();
    this.http.get('/api/model/' + id, { responseType: 'json' }).pipe(
      catchError((error) => {
        this.showError('Error archived data: ' + error.message);
        console.error('Error archived data:', error);
        this.hideLoading();
        return throwError(() => new Error('Error fetching schema'));
      })
    ).subscribe(async (_result) => {
      console.log("Received archived model data:", _result);
      const result: {model: any, group: string, classname: string, module: string, schema: any} =
        (_result as {model: any, group: string, classname: string, module: string, schema: any});
      console.log("Fetched archived model:", result);
      this.buildForm(result['schema']);
      this.model = result['model'];
      this.selected_group = result['group'];
      this.filter_tractions(this.selected_group);
      this.selected_traction = result['classname'];
      this.executed = true;

      this.hideLoading();
      this.run_id = id;
      this.stopWatching();
      this.logs = [];
      this.selectedTabIndex = 1;
      this.watching_interval_id = setInterval(() => {
        this.watchOutput()
      }, 2000);
    });
  }

  async deleteArchived(id: string) {
    console.log("Deleting archived ID:", id);
    id=id.replace(' ', ':');

    const response = await fetch('/api/delete/' + id, {method: "DELETE"});
    if (response.ok) {
      this.showError('Deleted archived traction logs: ' + id);
      this.fetch_archive();
      this.fetch_archive();
    } else {
      this.showError('Failed to delete archived traction logs: ' + id + ':' + response.statusText);
      this.fetch_archive();
    }
  }

  ngOnInit(field: FormlyFieldConfig) {
    this.traction_groups = [];
    this.tractions_data = [];
    this.tractions = [];
    this.showLoading();
    this.fetch_archive();

    this.http.get('/api/tractions', { responseType: 'json' })
    .pipe(
      catchError((error) => {
        this.showError('Error fetching schema: ' + error.message);
        console.error('Error fetching schema:', error);
        this.hideLoading();
        return throwError(() => new Error('Error fetching schema'));
      })
    )
    .subscribe(result=> {
      (result as any[]).forEach((item: any) => {
        this.traction_groups.push(item.name);
        if (item.type === 'object') {
          item.type = 'multischema';
        }
      });
      this.tractions_data = result;
      this.hideLoading();
    });
  }

  ngOnDestroy(): void {
    this.stopWatching();
  }



  constructor(private formlyJsonschema: FormlyJsonschema, private http: HttpClient) {
    this.fields = [];
  }

  async watchOutput() {
    const full_id = `traction-logs-${this.run_id}`;
    console.log(this);
    console.log("watch output", full_id);
    var res = null;
    try {
      res = await fetch(`/api/watch/${full_id}?last_id=${this.logsLastId}&count=50`);
    } catch (error) {
      this.info_bar_content = "Error connecting to server. Is the backend running?";
      return;
    } finally {
      this.info_bar_content = "";
    }

    const data = await res.json();
    console.log("Received", data);
    for (const msg of data.messages) {
      console.log('Received:', msg);
      if (msg.id == this.logsLastId) {
        // Skip if this is the last seen message
        continue;
      }
      this.logsLastId = msg.id; // Update last seen ID
      this.logs.push(msg.message);
    }
    console.log("last id", this.logsLastId);
  }

  stopWatching() {
    if (this.watching_interval_id) {
      clearInterval(this.watching_interval_id);
    }
  }

  async onSubmit(model: any) {
    console.log("old run id", this.run_id);
    if (this.run_id) {
      const parts = this.run_id.split(':')
      parts[2] = uuidv4(); // Update the run_id with a new UUID
      this.run_id = parts.join(':');

      console.log("new run id", this.run_id);
    } else {
      this.run_id = this.selected_group +":" + this.selected_traction + ":" + uuidv4();
    }

    console.log("Running model", model);
    const response = await fetch(
      '/api/run/'+ this.run_id + "/" + this.traction_modules[this.selected_traction] +
        '/' + this.selected_group + '/'+ this.selected_traction,
      {method: "POST",
       headers: {"Content-Type": "application/json"},
       body: JSON.stringify(model)});

    this.executed = true;
    this.selectedTabIndex = 1;
    if (!response.body) {
      return;
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    var output = "";

    this.fetch_archive();
    this.stopWatching();

    this.logs = [];
    this.watching_interval_id = setInterval(() => {
      this.watchOutput()
    }, 2000);
  }
}
