import { ApplicationConfig, provideZoneChangeDetection } from '@angular/core';
import { provideRouter } from '@angular/router';
import { provideFormlyCore } from '@ngx-formly/core'
import { withFormlyMaterial } from '@ngx-formly/material';
import { routes } from './app.routes';
import { HttpClientModule } from '@angular/common/http';

import { ObjectTypeComponent } from './object.type';
import { ArrayTypeComponent } from './array.type';
import { MultiSchemaTypeComponent } from './multischema.type';

export const appConfig: ApplicationConfig = {
  providers: [
    provideZoneChangeDetection({ eventCoalescing: true }), provideRouter(routes),
    provideFormlyCore([
      ...withFormlyMaterial(),
      {
         types: [
          { name: 'array', component: ArrayTypeComponent },
          { name: 'object', component: ObjectTypeComponent },
          { name: 'multischema', component: MultiSchemaTypeComponent },
        ],
        validationMessages: [
          { name: 'required', message: 'This field is required' }
        ],
      }
      ]
    ),
  ]
};
