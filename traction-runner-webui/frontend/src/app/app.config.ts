import { ApplicationConfig, provideZoneChangeDetection } from '@angular/core';
import { provideRouter } from '@angular/router';
import { provideFormlyCore } from '@ngx-formly/core'
import { withFormlyMaterial } from '@ngx-formly/material';
import { routes } from './app.routes';
import { HttpClientModule } from '@angular/common/http';
import { AbstractControl, ValidationErrors } from '@angular/forms';

import { ObjectTypeComponent } from './object.type';
import { ArrayTypeComponent } from './array.type';
import { MultiSchemaTypeComponent } from './multischema.type';


export function IntegerValidator(control: AbstractControl): ValidationErrors | null {
  const value = control.value;

  if (value == null || value === '') return null; // Let required handle empty

  if (!Number.isInteger(Number(value))) {
    return { integer: true };
  }

  return null;
}

export const appConfig: ApplicationConfig = {
  providers: [
    provideZoneChangeDetection({ eventCoalescing: true }), provideRouter(routes),
    provideFormlyCore([
      ...withFormlyMaterial(),
      {
        validators: [
          { name: 'integer', validation: IntegerValidator }
        ],
        types: [
          { name: 'array', component: ArrayTypeComponent },
          { name: 'object', component: ObjectTypeComponent },
          { name: 'multischema', component: MultiSchemaTypeComponent },
        ],
        validationMessages: [
          { name: 'required', message: 'This field is required' },
          { name: 'integer', message: 'Must be an integer' },
        ],
      }
    ]),
  ]
};
