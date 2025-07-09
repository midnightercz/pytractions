import { Component } from '@angular/core';
import { FieldType } from '@ngx-formly/core';
import { NgIf, NgFor } from '@angular/common';
import { FormlyField, FormlyValidationMessage } from '@ngx-formly/core';

import { marked } from 'marked';


@Component({
  selector: 'formly-object-type',
  template: `
    <div class="card mb-3">
      <div class="card-body">
        <legend *ngIf="props.label">{{ props.label }}</legend>
        <p *ngIf="props.description" [innerHTML]="translate_md_description(props.description)"></p>
        <div class="alert alert-danger" role="alert" *ngIf="showError && formControl.errors">
          <formly-validation-message [field]="field"></formly-validation-message>
        </div>
        <formly-field *ngFor="let f of field.fieldGroup" [field]="f"></formly-field>
      </div>
    </div>
  `,
  standalone: true,
  imports: [NgIf, FormlyField, FormlyValidationMessage, NgFor],
})
export class ObjectTypeComponent extends FieldType {
  translate_md_description(description: string) {
    const desc_lines = description.split('\n');
    if (desc_lines.length > 1 && desc_lines[0] == 'markdown') {
      return marked.parse(desc_lines.slice(1).join('\n'))
    }
    return description;
  }
}
