import { Component } from '@angular/core';
import { FieldArrayType } from '@ngx-formly/core';
import { NgIf, NgFor } from '@angular/common';
import { FormlyField, FormlyValidationMessage } from '@ngx-formly/core';

@Component({
  selector: 'formly-array-type',
  template: `
    <div class="card mb-3">
      <div class="card-body">
        <legend *ngIf="props.label">{{ props.label }}</legend>
        <p *ngIf="props.description">{{ props.description }}</p>
        <div class="d-flex flex-row-reverse">
          <button class="btn btn-primary" type="button" (click)="add()">+</button>
        </div>

        <div class="alert alert-danger" role="alert" *ngIf="showError && formControl.errors">
          <formly-validation-message [field]="field"></formly-validation-message>
        </div>

        <div *ngFor="let field of field.fieldGroup; let i = index" class="row align-items-start">
          <formly-field class="col" [field]="field"></formly-field>
          <div *ngIf="true" class="col-2 text-right">
            <button class="btn btn-danger" type="button" (click)="remove(i)">-</button>
          </div>
        </div>
      </div>
    </div>
  `,
  standalone: true,
  imports: [NgIf, FormlyField, FormlyValidationMessage, NgFor],
})
export class ArrayTypeComponent extends FieldArrayType {}

